def get_llm_type(llm_name: str) -> str:
    if llm_name.startswith("gpt"):
        return "openai"
    elif llm_name.startswith("deepseek"):
        return "deepseek"
    elif llm_name.startswith("claude"):
        return "anthropic"