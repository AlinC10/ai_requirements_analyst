# Done
# TODO: add a form of loading until the document is processed - DONE
# FIXME: remove welcome message after the document was uploaded - DONE by changing welcoming message to a chat message
# FIXME: can't write a prompt when uploading the document - DONE
# FIXME: modify rag system logic to delete from Chroma Database collection files that were deleted from the UI or
#  just ignore them in the similarity search. - DONE
# TODO: add token used to the response so that users know how many tokens they used - DONE
# FIXME: modify source name for the file to the file name:
#  eg: Documentul se referă la un apel public pentru propuneri, fiind oferit pentru scopuri informative de către Agenția
#  Belgiană pentru Cooperare Internațională (Enabel) și Ministerul Palestinian al Muncii (MoL). [Source:
#  tmp440dhrpy.pdf | Page: 0], [Source: tmpkdn8jliy.pdf | Page: 0], [Source: tmpm205bpg4.pdf | Page: 0],
#  [Source: tmpn1jdymeh.pdf | Page: 0], [Source: tmpn9vmlmby.pdf | Page: 0]. - DONE (testing)


# Testing
# TODO: add functions in the prompt: - partially DONE (needs more testing)
#  /classify [topic] - Extract & classify functional vs. non-functional requirements.
#  /defects [topic] - Deep audit for ambiguities, contradictions, or missing constraints.
#  /stories [topic] - Generate Agile User Stories with business value.
#  /usecases [topic] - Generate highly detailed Use Cases.
#  /criteria [topic] - Generate BDD Acceptance Criteria covering edge cases.
#  /diagram [topic] - Generate and render a Mermaid architecture/flow diagram.

# TODO: make LLM cite where he got the information from - DONE (needs testing)

# FIXME: add arguments for the retriever search, because sometimes it can retrieve a document that was not loaded in the
#  current session, which also happened in the example from above, where I uploaded the test.pdf and it gave information
#  related to other PDF. - SOLVE by always resetting the chroma db when enter a new session. When the system will hold
#  chats history, this will be need to be modified by searching with arguments.


# TODO
# TODO: implement a local llm mode, for documents that should not be publicly available

# FIXME: for one word, or just not long enough sentences it does not retrieve any context from the document

# TODO: refactor app.py add comments in the code for classes, methods, functions etc. so that it's easier to work for
#  future updates

# FIXME: remove print() used in the debugging process

# TODO: add conversations history so that users can select past conversation
#  and continue the discussion from where they left
