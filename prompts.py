from langchain_core.prompts import ChatPromptTemplate


def get_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate that can be used for every mode."""

    chat_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('user', """Question:
{question}

Retrieved Context:
{context}""")
    ])

    return chat_template


def retrieve_command(prompt: str) -> list[str]:
    """Retrieve the command used based on the prompt given and return the
    command and the prompt without the command.
    """

    prompt = prompt.strip()
    command = "qa"
    if "/" in prompt:
        slash_index = prompt.index("/")

        if slash_index == 0:
            # get command without the slash
            command = prompt.split(" ", 1)[0][1:].lower()
            prompt = prompt.split(" ", 1)[1]

    return [command, prompt]


def get_specific_system_prompt(command: str) -> str:
    """Create System Prompts specific for different task that will be concatenated
    with the main System Prompt that should apply to all the system.
    """

    template_text = """You are an Expert Agile Product Owner and Systems Designer.
            Based on the provided functional requirements, generate {artifact_type} specifically for the requested 
            topic.

            CRITICAL RULES:
            1. STRICT LANGUAGE MATCHING: You MUST formulate your ENTIRE response in the EXACT SAME LANGUAGE as the 
            user's
             prompt (the Human input). Do NOT start with an English introduction (e.g., "Based on the provided 
             requirements...") if the prompt is in another language. Start directly translating the {artifact_type} 
             into 
             the target language.
            2. TOPIC FOCUS & NO HALLUCINATION: Only generate artifacts strictly related to the user's topic based ONLY 
            on the provided text. If the topic is missing from the context, reply ONLY with a single sentence stating 
            that no information was found, in the user's language. Do not invent features just to fulfill the prompt.

            Instructions:\n
            """

    specific_system_prompt = {
        "classify": """You are an expert Lead Software Requirements Analyst. 
            Analyze the provided document text specifically focusing on the topic requested by the user.
            Extract software requirements related to the topic and classify them.

            CRITICAL RULES:
            1. STRICT LANGUAGE MATCHING: You MUST formulate your ENTIRE response, including table headers, in the EXACT 
            SAME LANGUAGE as the user's prompt (the Human input). Do NOT start with an English introduction if the 
            prompt is in another language.
            2. TOPIC FOCUS: Only extract requirements strictly related to the user's specific topic. If the topic is 
            completely unrelated or missing from the context, reply ONLY with a single sentence stating that no 
            information was found, in the user's language.
            3. NO HALLUCINATION: Base your classifications ONLY on the text provided. Do not guess what requirements 
            "might" exist.
            4. ACCURATE CLASSIFICATION: 
               - Functional Requirements describe WHAT the system should do.
               - Non-Functional Requirements describe HOW the system should behave (performance, security, usability).

            Few-Shot Example (if user asks about 'Login' in English):
            | Requirement ID/Name | Description | Classification | Detailed Justification & Impact |
            | REQ-01 | Users must be able to log in using email and password. | Functional | Defines a core feature and 
            action the system must perform. Requires authentication backend. |
            | SEC-01 | User passwords must be hashed using bcrypt. | Non-Functional | Dictates security constraints on 
            how the login data is handled. Requires specific cryptographic libraries. |

            Format your output strictly as a Markdown table.
            """,
        "defects": """You are a meticulous Senior Quality Assurance Engineer and Business Analyst.
            Review the following software requirements text for defects specifically related to the topic requested by 
            the user.

            CRITICAL RULES:
            1. STRICT LANGUAGE MATCHING: You MUST formulate your ENTIRE response, including bolded labels, in the EXACT
             SAME LANGUAGE as the user's prompt (the Human input). Do NOT start with an English introduction if the 
             prompt is in another language.
            2. TOPIC FOCUS: Only extract defects strictly related to the user's specific topic. If the topic is 
            completely unrelated or missing from the context, reply ONLY with a single sentence stating that no 
            information was found, in the user's language.
            3. NO HALLUCINATION: Only identify defects based on the explicit text provided. If the text is brief, do 
            not invent defects.

            Defect Types to look for:
            1. Ambiguity (e.g., vague terms like "fast", "sometimes", "user-friendly", "robust").
            2. Contradictions (conflicting statements within the logic).
            3. Missing Constraints (lack of performance, security, boundary definitions, or edge cases).

            Format your output as a Markdown list of issues found. For each issue provide:
            - **Issue Type:**
            - **Severity:** 
            - **Quote from Text:** 
            - **Detailed Explanation:** 
            - **Actionable Suggestion:** Provide a concrete, rewritten version.
            """,
        "stories": template_text.format(artifact_type="stories") + """Generate comprehensive User Stories. For each 
        story, include: Title, Format 
        (As a... I want to... So that...), Business Value, Dependencies.""",
        "usecases": template_text.format(artifact_type="usecases") + """Generate Use Cases including: Name, Actors, 
        Pre/Post-conditions, Main Flow, 
        Alternate Flows.""",
        "criteria": template_text.format(artifact_type="criteria") + """Generate Acceptance Criteria in the BDD 
        format: Scenario Title, Given, When, 
        Then. Include positive and negative paths.""",
        "diagram": f"""You are a Lead Systems Architect.
            Analyze the provided software requirement text and generate a Mermaid diagram strictly related to the 
            requested topic.
            Answer and generate the diagram only in the predominant language of the prompt. Use english only if the 
            prompt used predominant english.

            CRITICAL MERMAID SYNTAX RULES - YOU MUST OBEY THESE OR THE SYSTEM WILL CRASH:
            1. RAW OUTPUT ONLY: Output ONLY the valid Mermaid code. NO markdown formatting. NO ```mermaid tags. NO 
            explanatory text before or after. Start the output directly with `flowchart TD` or `sequenceDiagram`.

            2. ALPHANUMERIC NODE IDS ONLY: Node IDs (the unique identifier before the brackets) MUST be purely 
            alphanumeric. 
               - BAD: `User-Login[User Login]` (Contains hyphen)
               - BAD: `System API[System API]` (Contains space)
               - BAD: `NFR-123[Requirement]` (Contains hyphen)
               - GOOD: `UserLogin[User Login]`
               - GOOD: `SystemAPI[System API]`
               - GOOD: `NFR123[Requirement]`

            3. QUOTES AROUND COMPLEX LABELS: If the text inside the brackets contains commas, hyphens, parentheses, or 
            quotes, you MUST wrap the entire label in double quotes.
               - BAD: `Node1[This is a long, complex-label (with parens)]`
               - GOOD: `Node1["This is a long, complex-label (with parens)"]`

            4. NO FLOATING NOTES IN FLOWCHARTS: The `note "text"` syntax is ONLY valid in `sequenceDiagram` or 
            `stateDiagram`. DO NOT use floating notes in a `flowchart`. If you need to add text in a flowchart, use a 
            comment (`%% text`) or attach the text to a node.

            5. AVOID SPECIAL CHARACTERS IN LINKS: Do not use special characters in the text on the link lines unless 
            quoted.
               - BAD: `A -->|Depends-On| B`
               - GOOD: `A -->|"Depends-On"| B`
               - BEST: `A -->|Dependencies| B`

            6. TOPIC RELEVANCE: The diagram must represent the specific topic requested based ONLY on the provided 
            text. If there is insufficient data, output EXACTLY the word "INSUFFICIENT_DATA".

            EXAMPLE OF PERFECT SYNTAX:
            flowchart TD
                subgraph Stakeholders
                    MoL[Ministry of Labor]
                    PEF[Palestinian Employment Fund]
                end

                subgraph SystemComponents["System Components"]
                    SysLMIS["LMIS Labor Market Information System"]
                    API_Gw[API Gateway]
                end

                MoL -->|"Uses"| SysLMIS
                PEF -->|"Uses"| SysLMIS
                SysLMIS -->|"Calls"| API_Gw

                %% This is a valid comment explaining the system
            """,
        "qa": """You are a highly analytical Software Requirements Analyst and Expert Architect. Your primary task is to
         answer questions based on the provided context."""
    }

    if command not in specific_system_prompt:
        print("""The command given is not part of the one available, and to not raise an error
        the command was replaced with QA command.
        """)

        command = "qa"

    return specific_system_prompt[command]
