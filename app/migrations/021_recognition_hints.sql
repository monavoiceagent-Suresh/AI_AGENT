-- A persona's speech-recognition vocabulary (app/recognition.py): sent to the ASR provider as keyterms, never to
-- the LLM. Additive with an empty default, so every existing agent keeps plain recognition, exactly as before.
ALTER TABLE agents ADD COLUMN recognition_hints text NOT NULL DEFAULT '';
