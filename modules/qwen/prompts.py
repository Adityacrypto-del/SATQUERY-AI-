"""Prompt templates for Qwen2.5-VL remote-sensing analysis.

Each function returns a formatted prompt string. The prompts are
designed for Qwen2.5-VL-Instruct models and explicitly distinguish
temporal order (T1 = earlier, T2 = later) for bi-temporal tasks.
"""


def single_image_prompt(query: str) -> str:
    """Build a prompt for single-image remote-sensing analysis.

    Parameters
    ----------
    query:
        The user's natural-language question about the image.
    """
    return (
        "You are a remote-sensing image analysis assistant.\n\n"
        "Analyze the provided satellite or aerial image and answer "
        "the following question:\n\n"
        f"Question: {query}\n\n"
        "Provide:\n"
        "1. A detailed observation of what you see in the image.\n"
        "2. Any notable objects, structures, or land-cover types.\n"
        "3. Your confidence level (low / medium / high) in your "
        "analysis.\n\n"
        "Be specific about spatial features. Do not invent "
        "measurements you cannot determine from the image."
    )


def bitemporal_prompt(query: str) -> str:
    """Build a prompt for bi-temporal change analysis.

    The first image is T1 (earlier observation) and the second
    image is T2 (later observation).

    Parameters
    ----------
    query:
        The user's natural-language question about temporal change.
    """
    return (
        "You are a remote-sensing change detection assistant.\n\n"
        "You are given two images of the same geographic area "
        "taken at different times:\n"
        "- Image 1 (T1): The EARLIER observation.\n"
        "- Image 2 (T2): The LATER observation.\n\n"
        "Analyze both images and answer the following question:\n\n"
        f"Question: {query}\n\n"
        "Provide:\n"
        "1. What you observe in the earlier image (T1).\n"
        "2. What you observe in the later image (T2).\n"
        "3. What changes occurred between T1 and T2.\n"
        "4. Your confidence level (low / medium / high) in your "
        "analysis.\n\n"
        "Important:\n"
        "- T1 is always the earlier image. T2 is always the later "
        "image.\n"
        "- Do not confuse the temporal order.\n"
        "- Do not invent precise measurements (pixel counts, areas) "
        "that you cannot determine visually.\n"
        "- Focus on what is visible and describe changes clearly."
    )


def final_reasoning_prompt(
    query: str,
    qwen_observation: str,
    specialist_summary: str,
    measurements: str,
) -> str:
    """Build a prompt for the final reasoning pass.

    The final reasoner receives the original query, the
    independent Qwen analysis, and the specialist evidence.

    Parameters
    ----------
    query:
        The user's original question.

    qwen_observation:
        The independent Qwen VLM observation from the first pass.

    specialist_summary:
        Structured evidence from the specialist pipeline.

    measurements:
        Quantitative measurements from the specialist pipeline.
    """
    return (
        "You are a remote-sensing analysis assistant producing a "
        "final answer.\n\n"
        "You have received:\n"
        "1. The user's question.\n"
        "2. Your own independent visual analysis.\n"
        "3. Quantitative evidence from a specialist pipeline.\n\n"
        f"User Question: {query}\n\n"
        f"Your Independent Analysis:\n{qwen_observation}\n\n"
        f"Specialist Evidence:\n{specialist_summary}\n\n"
        f"Specialist Measurements:\n{measurements}\n\n"
        "Instructions:\n"
        "- Use the specialist measurements as authoritative for "
        "any quantitative claims (pixel counts, areas, "
        "percentages).\n"
        "- Do NOT invent or override specialist measurements.\n"
        "- Use your visual analysis to provide context and "
        "interpretation.\n"
        "- If your analysis disagrees with the specialist, "
        "mention the disagreement and explain possible reasons.\n"
        "- Answer the user's actual question directly.\n"
        "- State your confidence level.\n"
        "- Mention any limitations or uncertainties."
    )
