# TODO: add a form of loading until the document is processed

# TODO: add functions in the prompt:
#  /classify [topic] - Extract & classify functional vs. non-functional requirements.
#  /defects [topic] - Deep audit for ambiguities, contradictions, or missing constraints.
#  /stories [topic] - Generate Agile User Stories with business value.
#  /usecases [topic] - Generate highly detailed Use Cases.
#  /criteria [topic] - Generate BDD Acceptance Criteria covering edge cases.
#  /diagram [topic] - Generate and render a Mermaid architecture/flow diagram.

# TODO: add token used to the response so that users know how many tokens they used
#  This can be a feature that users can deactivate from a settings menu.

# TODO: refactor app.py add comments in the code for classes,
#  methods, functions etc. so that it's easier to work for future
#  updates

# FIXME: remove print() used in the debugging process

# TODO: move data from .env to secrets.toml to upload the page to
#  Streamlit community

# FIXME: modify rag system logic to delete from Chroma Database collection
#  files that were deleted from the UI or just ignore them in the
#  similarity search.

# TODO: add conversations history so that users can select past conversation
#  and continue the discussion from where they left