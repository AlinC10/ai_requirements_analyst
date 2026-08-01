# Testing
# TODO: add functions in the prompt: - partially DONE (needs more testing)
#  /classify [topic] - Extract & classify functional vs. non-functional requirements.
#  /defects [topic] - Deep audit for ambiguities, contradictions, or missing constraints.
#  /stories [topic] - Generate Agile User Stories with business value.
#  /usecases [topic] - Generate highly detailed Use Cases.
#  /criteria [topic] - Generate BDD Acceptance Criteria covering edge cases.
#  /diagram [topic] - Generate and render a Mermaid architecture/flow diagram.

# TODO: implement a local llm mode, for documents that should not be publicly available - testing

# TODO: make LLM cite where he got the information from - DONE (needs testing)

# FIXME: add arguments for the retriever search, because sometimes it can retrieve a document that was not loaded in the
#  current session, which also happened in the example from above, where I uploaded the test.pdf and it gave information
#  related to other PDF. - SOLVE by always resetting the chroma db when enter a new session. When the system will hold
#  chats history, this will be need to be modified by searching with arguments.


# TODO
# FIXME: for one word, or just not long enough sentences it does not retrieve any context from the document

# TODO: refactor app.py add comments in the code for classes, methods, functions etc. so that it's easier to work for
#  future updates

# FIXME: remove print() used in the debugging process

# TODO: add conversations history so that users can select past conversation
#  and continue the discussion from where they left

# TODO: improved error handling. enhance error messages and provide more guidance to the user.
# TODO: add more feedback to the user for the moment when Ollama downloads a new model, such as a progress bar or
#  more explicit status messages.
