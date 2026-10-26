import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from groq import Groq
from config.settings import settings


class AgentInterface:
    """Invoke chat models through configured providers with ordered fallback."""

    def __init__(self):
        """Load credentials, provider priority, logging, and the Groq client."""
        self.providers = [
            ("openai/gpt-oss-120b", "groq"),
            ("gemini-3.8-flash", "gemini"),
            ("gemini-3.7-flash", "gemini"),
            ("gemini-3.6-flash", "gemini"),
            ("gemini-3.5-flash", "gemini"),
            ("openai/gpt-oss-20b", "groq")
        ]

        self.GROQ_API_KEY = settings.GROQ_API_KEY
        self.GEMINI_API_KEY = settings.GEMINI_API_KEY

        self.logger = logging.getLogger(__name__)

        self.tools = [
        ]

        self.groq_client = Groq(
            api_key=self.GROQ_API_KEY
        )

    def _get_llm(self, model="gemini-3.8-flash", provider="gemini"):
        """Create a LangChain chat model for a configured provider.

        Args:
            model: Provider-specific model identifier.
            provider: Provider key, currently ``gemini`` or ``groq``.

        Raises:
            ValueError: If the provider is not supported.
        """
        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=model,
                api_key=self.GEMINI_API_KEY
            )

        elif provider == "groq":
            return ChatGroq(
                model=model,
                api_key=self.GROQ_API_KEY
            )

        raise ValueError(f"Provider {provider} not implemented")


    def invoke(self, user_query, system_prompt=None):
        """Invoke an agent, trying providers in priority order.

        Args:
            user_query: User content sent to the model.
            system_prompt: Optional system-level instruction.

        Returns:
            The LangChain agent response.

        Raises:
            RuntimeError: If every configured provider fails.
        """
        provider_index = 0

        while provider_index < len(self.providers):

            model, provider = self.providers[provider_index]

            try:
                llm = self._get_llm(model, provider)

                agent = create_agent(
                    llm,
                    system_prompt=system_prompt,
                    tools=self.tools,
                )

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_query
                        }
                    ]
                })

                self.logger.info(
                    f"LLM request successful: {provider} / {model}"
                )

                return response

            except Exception as e:
                self.logger.warning(
                    f"LLM failed: {provider} / {model}: {e}"
                )

                provider_index += 1

        raise RuntimeError("All configured LLM providers failed")

    def structured_invoke(self, user_query, response_schema, system_prompt=None):
        """Return a provider response validated against a Pydantic schema.

        Groq uses its JSON-schema response mode; other providers use
        LangChain's structured-output interface. Provider failures are logged
        and retried using the next configured model.

        Args:
            user_query: User content sent to the model.
            response_schema: Pydantic model class used to validate the result.
            system_prompt: Optional system-level instruction.

        Returns:
            An instance of ``response_schema``.

        Raises:
            RuntimeError: If every configured provider fails validation or invocation.
        """
        provider_index = 0

        while provider_index < len(self.providers):

            model, provider = self.providers[provider_index]

            try:

                if provider == "groq":

                    schema = response_schema.model_json_schema()

                    schema["additionalProperties"] = False

                    response = self.groq_client.chat.completions.create(
                        model=model,
                        messages=(
                            ([{"role": "system", "content": system_prompt}]
                             if system_prompt else [])
                            + [{"role": "user", "content": user_query}]
                        ),
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_response",
                                "strict": True,
                                "schema": schema
                            }
                        }
                    )

                    content = response.choices[0].message.content

                    result = response_schema.model_validate(
                        json.loads(content)
                    )

                else:
                    llm = self._get_llm(model, provider)

                    structured_llm = llm.with_structured_output(
                        response_schema
                    )

                    llm_input = (
                        [
                            ("system", system_prompt),
                            ("human", user_query),
                        ]
                        if system_prompt
                        else user_query
                    )
                    result = structured_llm.invoke(llm_input)

                self.logger.info(
                    f"Structured LLM request successful: "
                    f"{provider} / {model}"
                )

                return result

            except Exception as e:

                self.logger.warning(
                    f"Structured LLM failed: "
                    f"{provider} / {model}: {e}"
                )

                provider_index += 1

        raise RuntimeError(
            "All configured LLM providers failed for structured output"
        )