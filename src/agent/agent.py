import os
import re
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from datetime import datetime
class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop.
    Uses tools to answer user queries through iterative reasoning.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = {t['name']: t for t in tools}
        self.max_steps = max_steps

    def get_system_prompt(self) -> str:
        """Build the system prompt with available tools and ReAct format instructions."""
        tool_descriptions = "\n".join(
            [f"- {name}: {t['description']}" for name, t in self.tools.items()]
        )
        current_year = datetime.now().year
        return (
            "You are a helpful shopping assistant for an electronics store. "
            "You can use the following tools to help answer the user's question:\n"
            f"{tool_descriptions}\n\n"
            "You MUST follow this EXACT format for every reasoning step:\n\n"
            "Thought: <your reasoning about what to do next>\n"
            "Action: <tool_name>(<argument>)\n\n"
            "After you receive an Observation, continue with another Thought/Action "
            "if needed, or provide your final answer using:\n\n"
            "Final Answer: <your complete answer to the user>\n\n"
            "IMPORTANT RULES:\n"
            "- Always start with a Thought before taking any Action.\n"
            "- Use EXACTLY one Action per step.\n"
            "- Do NOT invent information. Only use data from Observations.\n"
            "- When you have enough info, you MUST output 'Final Answer:'.\n"
            "- If no tool is needed, go directly to Final Answer.\n"
            "- IMPORTANT: Always respond to the user in the SAME language they spoke to you in.\n"
            "- IMPORTANT: You represent THIS store. NEVER advise the user to buy from other shops. Do NOT mention competitor store names (like Shopdunk, XTmobile, Oneway, v.v.) and do NOT quote their prices. Use online search ONLY to describe the product specs and general info.\n"
            f"- IMPORTANT: The current year is {current_year}. Do NOT claim a product hasn't been released just because it's past your knowledge cutoff. Rely purely on observation results.\n"
        )

    def run(self, user_input: str) -> str:
        """
        Execute the ReAct loop:
        1. Generate Thought + Action from the LLM.
        2. Parse Action, execute the tool, get Observation.
        3. Append Observation to the scratchpad and repeat.
        4. Return the Final Answer when found, or a fallback after max_steps.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        
        system_prompt = self.get_system_prompt()
        scratchpad = f"User: {user_input}\n"
        steps = 0

        while steps < self.max_steps:
            steps += 1
            logger.log_event("AGENT_STEP", {"step": steps})

            # Ask the LLM to continue the reasoning
            result = self.llm.generate(scratchpad, system_prompt=system_prompt)
            llm_output = result["content"]
            scratchpad += llm_output + "\n"

            # Parse Thought from the LLM output to print it
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", llm_output, re.DOTALL)
            if thought_match:
                print(f"💡 Thought: {thought_match.group(1).strip()}")

            # Check for Final Answer
            final_answer = self._parse_final_answer(llm_output)
            if final_answer:
                logger.log_event("AGENT_END", {"steps": steps, "status": "final_answer", "response": final_answer})
                return final_answer

            # Parse and execute Action
            action_name, action_arg = self._parse_action(llm_output)
            if action_name:
                print(f"🛠️  Action: {action_name}({action_arg})")
                observation = self._execute_tool(action_name, action_arg)
                print(f"🔍 Observation: {observation}")
                scratchpad += f"Observation: {observation}\n"
                logger.log_event("TOOL_RESULT", {
                    "tool": action_name,
                    "args": action_arg,
                    "result_preview": str(observation)[:200]
                })
            else:
                # No action and no final answer — nudge the LLM
                scratchpad += (
                    "Observation: You did not provide a valid Action or Final Answer. "
                    "Please follow the format: Action: tool_name(argument) or Final Answer: ...\n"
                )

        # Exhausted max steps — ask the LLM for a summary
        scratchpad += "Thought: I have reached the maximum number of steps. I must give my Final Answer now.\nFinal Answer:"
        result = self.llm.generate(scratchpad, system_prompt=system_prompt)
        final_text = result["content"].strip()
        final_answer = self._parse_final_answer(final_text)

        answer_to_return = final_answer if final_answer else final_text
        
        # Super-safe fallback: violently strip 'Final Answer:' and any leftover thoughts/actions
        answer_to_return = re.sub(r"(?i)\**Final\s*Answer:\**\s*", "", answer_to_return)
        answer_to_return = re.sub(r"(?im)^Thought:.*?$", "", answer_to_return)
        answer_to_return = re.sub(r"(?im)^Action:.*?$", "", answer_to_return)
        answer_to_return = answer_to_return.strip()

        logger.log_event("AGENT_END", {"steps": steps, "status": "max_steps_reached", "response": answer_to_return})
        return answer_to_return

    def _parse_action(self, text: str) -> tuple:
        """
        Extract tool name and argument from text like:
            Action: search_product(iPhone)
        Returns (tool_name, argument) or (None, None).
        """
        match = re.search(r"Action:\s*(\w+)\((.*?)\)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip().strip("'\"")
        return None, None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from the LLM output."""
        # Use regex to catch case variations and markdown like **Final Answer:**
        match = re.search(r"(?i)\*?\*?Final Answer:\*?\*?\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """Look up and call the tool function by name."""
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
        try:
            return tool["func"](args)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"