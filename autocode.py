import os
import re
import subprocess
import json
import streamlit as st
from together import Together
from pathlib import Path

# Configuration
PROJECT_DIR = "together_ai_projects"
os.makedirs(PROJECT_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(PROJECT_DIR, "conversation_memory.json")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MAX_FIX_ATTEMPTS = 5  # Maximum number of automatic fix attempts

class StreamlitAutoCoder:
    def __init__(self):
        self.current_file = os.path.join(PROJECT_DIR, "generated_code.py")
        self.current_code = ""
        self.client = None
        self.conversations_dir = os.path.join(PROJECT_DIR, "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)
        self.current_conversation = "default"
        self.setup_session_state()
        self.load_memory()
        self.setup_ui()

    def setup_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False
        if "code_history" not in st.session_state:
            st.session_state.code_history = []
        if "current_conversation" not in st.session_state:
            st.session_state.current_conversation = "default"
        if "conversations" not in st.session_state:
            st.session_state.conversations = ["default"]

    def load_memory(self):
        """Load conversation memory from the selected conversation."""
        try:
            memory_file = os.path.join(self.conversations_dir, f"{self.current_conversation}.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    st.session_state.messages = data.get("messages", [])
                    st.session_state.code_history = data.get("code_history", [])
                    if st.session_state.messages:
                        last_code = next((msg["content"] for msg in reversed(st.session_state.messages) 
                                          if msg["type"] == "code"), "")
                        if last_code:
                            self.current_code = self._add_main_block(last_code)
            else:
                st.session_state.messages = []
                st.session_state.code_history = []
        except Exception as e:
            st.error(f"Failed to load memory: {str(e)}")

    def save_memory(self):
        """Save current conversation to the selected conversation file."""
        try:
            memory_file = os.path.join(self.conversations_dir, f"{self.current_conversation}.json")
            with open(memory_file, 'w') as f:
                json.dump({
                    "messages": st.session_state.messages,
                    "code_history": st.session_state.code_history
                }, f)
        except Exception as e:
            st.error(f"Failed to save memory: {str(e)}")

    def setup_ui(self):
        st.set_page_config(page_title="AI Python Coder", page_icon="🐍")
        self.setup_sidebar()
        self.setup_main_interface()

    def setup_sidebar(self):
        with st.sidebar:
            st.title("⚙️ Settings")
            st.session_state.api_key = st.text_input(
                "Together.ai API Key",
                type="password",
                value=st.session_state.api_key,
                help="Get your API key from together.ai"
            )
            
            st.session_state.dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
            
            if st.button("🔄 Clear Chat"):
                st.session_state.messages = []
                st.session_state.code_history = []
                self.current_code = ""
                self.save_memory()
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Conversations")
            for conversation in st.session_state.conversations:
                col1, col2 = st.columns([8, 1])
                with col1:
                    if st.button(conversation, key=f"select_{conversation}"):
                        self.switch_conversation(conversation)
                with col2:
                    if st.button("🗑️", key=f"delete_{conversation}"):
                        self.delete_conversation(conversation)
            
            if st.button("➕ New Conversation"):
                self.create_new_conversation()
            
            st.markdown("---")
            st.markdown("### Code History")
            for i, code in enumerate(reversed(st.session_state.code_history[-5:])):
                if st.button(f"Version {len(st.session_state.code_history)-i}", key=f"hist_{i}"):
                    self.current_code = code
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Loaded code version {len(st.session_state.code_history)-i}",
                        "type": "system"
                    })
                    st.rerun()

    def create_new_conversation(self):
        """Create a new conversation and switch to it."""
        if st.session_state.messages:
            # Generate a short name based on the first user input
            first_prompt = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "New Conversation")
            new_conversation_name = self._generate_short_name(first_prompt)
        else:
            new_conversation_name = f"Conversation {len(st.session_state.conversations) + 1}"
        
        if new_conversation_name in st.session_state.conversations:
            st.warning("Conversation name already exists. Creating a unique name.")
            new_conversation_name = f"{new_conversation_name}_{len(st.session_state.conversations) + 1}"
        
        st.session_state.conversations.append(new_conversation_name)
        st.session_state.current_conversation = new_conversation_name
        self.current_conversation = new_conversation_name
        st.session_state.messages = []
        st.session_state.code_history = []
        self.save_memory()
        st.success(f"Created and switched to new conversation: {new_conversation_name}")
        st.rerun()

    def delete_conversation(self, conversation_name):
        """Delete a conversation."""
        if conversation_name in st.session_state.conversations:
            st.session_state.conversations.remove(conversation_name)
            conversation_file = os.path.join(self.conversations_dir, f"{conversation_name}.json")
            if os.path.exists(conversation_file):
                os.remove(conversation_file)
            st.success(f"Deleted conversation: {conversation_name}")
            if st.session_state.current_conversation == conversation_name:
                st.session_state.current_conversation = "default"
                self.current_conversation = "default"
                self.load_memory()
            self.save_memory()
            st.rerun()
        else:
            st.error("Conversation not found.")

    def _generate_short_name(self, prompt):
        """Generate a short name for the conversation based on the first input prompt."""
        words = prompt.split()
        return "-".join(words[:3]).capitalize() if len(words) > 3 else prompt.capitalize()

    def switch_conversation(self, conversation_name):
        """Switch to a different conversation."""
        if conversation_name in st.session_state.conversations:
            st.session_state.current_conversation = conversation_name
            self.current_conversation = conversation_name
            self.load_memory()
            st.success(f"Switched to conversation: {conversation_name}")
        else:
            st.error("Conversation not found.")
        # Ensure buttons are displayed based on the current code state
        self.setup_main_interface()

    def create_conversation(self, conversation_name):
        """Create a new conversation."""
        if conversation_name and conversation_name not in st.session_state.conversations:
            st.session_state.conversations.append(conversation_name)
            st.session_state.current_conversation = conversation_name
            self.current_conversation = conversation_name
            self.save_memory()
            st.success(f"Created new conversation: {conversation_name}")
            st.rerun()

    def modify_code_with_prompt(self, prompt):
        """Modify the current code based on the user's prompt and continue the conversation."""
        if not self.current_code:
            st.error("No code to modify. Please generate code first.")
            return
        
        with st.spinner("Modifying code..."):
            modify_prompt = f"""
Please modify the following Python code based on the user's request.

Original Code:
{self.current_code}

User's Request:
{prompt}

Additional instructions:
1. Provide the complete modified code.
2. Maintain the original functionality unless specified otherwise.
3. Follow PEP 8 guidelines.

Return ONLY the modified Python code without any explanations or markdown formatting.
"""
            response = self.generate_code(modify_prompt)
            
            if "error" in response:
                st.error(response["error"])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Error: {response['error']}", 
                    "type": "error"
                })
                return
            
            # Update the current code with the modified version
            self.current_code = self._add_main_block(response["code"])
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "type": "text"
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["code"], 
                "type": "code"
            })
            st.session_state.code_history.append(self.current_code)
            self.save_memory()
            st.success("Code modified successfully!")
            st.code(self.current_code, language="python")

    def setup_main_interface(self):
        st.title("🐍 AI Python Coder")
        st.caption(f"Powered by {MODEL_NAME}")
        
        # Apply dark mode if enabled
        if st.session_state.dark_mode:
            self.apply_dark_mode()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.container():  # Use container to group chat messages
                if message["role"] == "user":
                    st.markdown(f"**User:** {message['content']}")
                elif message["role"] == "assistant":
                    if message["type"] == "code":
                        st.code(message["content"], language="python")
                    else:
                        st.markdown(f"**Assistant:** {message['content']}")
                elif message["role"] == "system":
                    st.markdown(f"**System:** {message['content']}")

        # Chat input
        prompt = st.text_input("Describe the code you want to generate or modify...")
        if prompt:
            col1, col2, col3 = st.columns([1, 1, 1])
            if not self.current_code:
                # Automatically generate code for the first prompt
                self.process_user_input(prompt)
            else:
                # Show options for "Generate New Code," "Modify Existing Code," and "Stop Generating"
                with col1:
                    if st.button("Generate New Code"):
                        self.process_user_input(prompt)
                with col2:
                    if st.button("Modify Existing Code"):
                        self.modify_code_with_prompt(prompt)
                with col3:
                    if st.button("⏹ Stop Generating"):
                        st.warning("Code generation stopped by the user.")
                        st.stop()

    def apply_dark_mode(self):
        dark_mode_css = """
        <style>
            .stApp {
                background-color: #0E1117;
                color: white;
            }
            .stChatInput textarea {
                color: white !important;
            }
            .stCodeBlock pre {
                background-color: #1E1E1E;
            }
        </style>
        """
        st.markdown(dark_mode_css, unsafe_allow_html=True)

    def process_user_input(self, prompt):
        # Add user message to chat
        with st.container():
            st.markdown(f"**User:** {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        
        # Initialize client if needed
        if not self.client and st.session_state.api_key:
            try:
                self.client = Together(api_key=st.session_state.api_key)
            except Exception as e:
                st.error(f"Failed to initialize client: {str(e)}")
                return
        
        # Generate code
        with st.container():
            st.markdown("**Assistant is generating code...**")
            with st.spinner("Generating code..."):
                response = self.generate_code(prompt)
                
                if "error" in response:
                    st.error(response["error"])
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Error: {response['error']}", 
                        "type": "error"
                    })
                    # Add a retry button
                    if st.button("🔄 Retry"):
                        self.process_user_input(prompt)
                    return
                
                # Display the generated code
                st.code(response["code"], language="python")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["code"], 
                    "type": "code"
                })
                
                # Save the generated code to history
                self.current_code = self._add_main_block(response["code"])
                st.session_state.code_history.append(self.current_code)
                self.save_memory()
                
                # Automatically run the generated code
                self.run_and_fix_code()

    def run_and_fix_code(self):
        """Run the current code and automatically fix errors if they occur."""
        if not self.current_code:
            st.error("No code to run")
            return
        
        with st.spinner("Running code..."):
            output, success = self._execute_code(self.current_code)
        
        if success:
            st.success("Code executed successfully!")
            st.code(output, language="text")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Execution successful:\n{output}",
                "type": "execution"
            })
        else:
            st.error("Code execution failed. Attempting to fix...")
            st.code(output, language="text")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Execution failed:\n{output}",
                "type": "execution"
            })
            self.fix_current_code()

    def generate_code(self, prompt: str) -> dict:
        # Include previous messages for context
        messages = [{"role": "system", "content": """You are an expert Python developer. Generate complete, functional Python code that:
1. Follows best practices and PEP 8 guidelines
2. Includes proper error handling
3. Has clear comments where needed
4. Is ready to execute with appropriate usage examples

Return ONLY the Python code without any explanations or markdown formatting."""}]
        
        # Add relevant previous messages for context (last 3 exchanges)
        prev_messages = [msg for msg in st.session_state.messages[-6:] if msg["type"] in ["text", "code"]]
        for msg in prev_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            generated_code = response.choices[0].message.content
            return {
                "code": self._clean_code(generated_code),
                "raw_response": generated_code
            }
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}  # Fixed mismatched parenthesis

    def _clean_code(self, code: str) -> str:
        """Remove any non-code formatting from the response."""
        code = re.sub(r'```python\n?|\n?```', '', code)
        return code.strip()

    def _add_main_block(self, code: str) -> str:
        """Ensure the code has a main execution block if needed."""
        if "__name__" not in code:
            if "def main(" in code:
                code += "\n\nif __name__ == '__main__':\n    main()"
            else:
                code += "\n\nif __name__ == '__main__':\n    pass"
        return code

    def run_current_code(self):
        """Run the current code and display the output."""
        if not self.current_code:
            st.error("No code to run")
            return
        with st.spinner("Running code..."):
            output, success = self._execute_code(self.current_code)
        if success:
            st.success("Code executed successfully!")
            st.code(output, language="text")
        else:
            st.error("Code execution failed")
            st.code(output, language="text")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Execution {'successful' if success else 'failed'}:\n{output}",
            "type": "execution"
        })
        self.save_memory()

    def _execute_code(self, code: str) -> tuple:
        """Execute the code and return output and success status."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.current_file), exist_ok=True)
            
            with open(self.current_file, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Use the same Python executable that's running the script
            python_exec = "python3" if os.name != 'nt' else "python"
            
            result = subprocess.run(
                [python_exec, self.current_file],
                capture_output=True,
                text=True,
                timeout=15
            )
            return (result.stdout, True) if result.returncode == 0 else (result.stderr, False)
        except subprocess.TimeoutExpired:
            return ("⏰ Timeout: Possible infinite loop", False)
        except Exception as e:
            return (f"🚨 Execution Error: {str(e)}", False)

    def save_current_code_to_history(self):
        """Save the current code to the code history."""
        if not self.current_code:
            st.error("No code to save")
            return
        st.session_state.code_history.append(self.current_code)
        self.save_memory()
        st.success("Code saved to history")

    def save_current_code(self):
        if not self.current_code:
            st.error("No code to save")
            return
        
        with st.container():
            filename = st.text_input("Enter filename (without .py):", "generated_code", key="filename_input")
            if st.button("Confirm Save", key="confirm_save"):
                if filename:
                    save_path = os.path.join(PROJECT_DIR, f"{filename}.py")
                    try:
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(self.current_code)
                        st.success(f"Code saved to {save_path}")
                        st.session_state.messages.append({
                            "role": "system", 
                            "content": f"Code saved to {save_path}", 
                            "type": "system"
                        })
                        st.session_state.code_history.append(self.current_code)
                        self.save_memory()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save file: {str(e)}")

    def fix_current_code(self):
        if not self.current_code:
            st.error("No code to fix")
            return

        # Get the last execution error if available
        last_error = ""
        for msg in reversed(st.session_state.messages):
            if msg["type"] == "execution" and "failed" in msg["content"]:
                last_error = msg["content"]
                break

        # Container to show all fixes in one expandable section
        with st.expander("🔧 Auto-Fix Progress", expanded=True):
            fix_container = st.empty()
            
            attempt = 1
            fixed_successfully = False
            current_fix_code = self.current_code
            
            while attempt <= MAX_FIX_ATTEMPTS and not fixed_successfully:
                with fix_container.container():
                    st.write(f"⚙️ Fix attempt {attempt}/{MAX_FIX_ATTEMPTS}")
                    
                    fix_prompt = self._build_fix_prompt(current_fix_code, last_error)
                    
                    with st.spinner(f"Generating fix {attempt}..."):
                        response = self.generate_code(fix_prompt)
                    
                    if "error" in response:
                        st.error(f"Fix generation failed: {response['error']}")
                        break
                    
                    fixed_code = self._add_main_block(response["code"])
                    st.code(fixed_code, language="python")
                    
                    # Test the fixed code
                    with st.spinner("Testing fixed code..."):
                        output, success = self._execute_code(fixed_code)
                    
                    if success:
                        st.success("✅ Code fixed and runs successfully!")
                        st.code(output, language="text")
                        fixed_successfully = True
                        current_fix_code = fixed_code
                    else:
                        st.error(f"❌ Fix {attempt} still has errors")
                        st.code(output, language="text")
                        last_error = output
                        current_fix_code = fixed_code
                        attempt += 1
            
            if fixed_successfully:
                # Update the current code with the successful fix
                self.current_code = current_fix_code
                st.session_state.messages.extend([
                    {
                        "role": "assistant",
                        "content": f"Fixed code after {attempt} attempts",
                        "type": "system"
                    },
                    {
                        "role": "assistant",
                        "content": current_fix_code,
                        "type": "code"
                    },
                    {
                        "role": "assistant",
                        "content": f"Execution successful:\n{output}",
                        "type": "execution"
                    }
                ])
                st.session_state.code_history.append(current_fix_code)
                self.save_memory()
                st.balloons()
            else:
                st.error(f"Unable to fix after {MAX_FIX_ATTEMPTS} attempts. Please try modifying your request.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Failed to fix after {MAX_FIX_ATTEMPTS} attempts",
                    "type": "error"
                })

    def _build_fix_prompt(self, code: str, error: str = "") -> str:
        """Construct a comprehensive fix prompt with context"""
        prompt = f"""
Please fix this Python code. Analyze the code carefully and provide a complete corrected version 
that addresses all syntax errors, runtime errors, and logical issues.

The code to fix:
{code}
"""
        if error:
            prompt += f"""
The error encountered when running the code:
{error}
"""
        prompt += """
Additional instructions:
1. Provide the complete fixed code, not just patches
2. Include comments explaining the fixes
3. Maintain the original functionality
4. Ensure proper error handling
5. Follow PEP 8 guidelines

Return ONLY the fixed Python code without additional explanations.
"""
        return prompt.strip()

if __name__ == "__main__":
    app = StreamlitAutoCoder()