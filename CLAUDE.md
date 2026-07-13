# Claude Brain — Mason's Global Rules

---

## Standard Rules

- Before changing any existing code, **ask first** — let Mason verify before anything is modified.
- Every function must have a one-sentence description of what it does.
- Make every function as efficient as possible.
- Be able to explain any function at a deeper level when asked.
- When debugging, go through the code thoroughly — make sure no bugs remain and all edge cases are covered.
- Never create files outside the active project folder.
- Respond in a detailed but concise manner. Follow instructions exactly.

---

## Coding Style

- Use descriptive variable names throughout.
- Lines must be no longer than **88 characters**.
- Code must look consistent and easy to read from top to bottom.
- Optimize for efficiency and readability as you write, not after.

---

## Python-Specific Style

- Follow **PEP 8** formatting at all times.
- Add a one-line comment before each function describing what it does.
- Use descriptive variable names.
- Format long dictionaries neatly across multiple lines for readability.
- **OOP:** When creating classes and their objects, put them in a separate `.py` file. Separate each class/object with a divider.
- Before starting a project, list all libraries that will be used and flag which ones are not yet installed.

---

## Backend Rules

- **Never hard-code API keys.** Always use `.env` files and environment variables. Keep them separate and secure.
- **Web development:**
  - Always sanitize and validate user inputs.
  - Use authentication middleware unless explicitly told otherwise.
  - If user input looks like code, reject it — do not let it pass through.
  - Always build a middleware layer for an extra layer of security.
- Backend languages in use: **Python**, **Java**, **C++**

---

## Frontend / Backend Translation

- When connecting a frontend to a backend, always create a dedicated **Translator** file.
- Name it `Translator` and write it in the **backend language**.
- The Translator handles all communication between frontend and backend — do not mix concerns.

---

## Commit Rules

- Only commit a file when it is **fully functional with no bugs**.
- **Never commit `.env` files or API keys** under any circumstances.
- Before committing, give Mason a brief summary of what is being committed and **wait for approval**.

---

## Testing Rules

- Test files in isolation before integrating them into the main codebase.
- Cover all edge cases during testing.
- Thoroughly verify the code runs correctly before marking it done.

---

## Session Startup Checklist

At the start of every session:
1. Load these rules — treat them as non-negotiable.
2. Identify the active project folder (never create files outside it).
3. Note the language/framework in use so style rules apply correctly.
4. Restate the user's request in one sentence before acting.
5. Ask before changing any existing code.
6. Surface assumptions instead of guessing.

End of session: summarize what was done, what's pending, and any open questions.



---

## Agents (available for any project)

**Code Reviewer** — Debugs code, finds and fixes all errors.

**Security** — Ensures no API keys are leaked, no security breaches exist. Make the application as secure as possible.

**UI/UX Designer** — Designs sleek, cohesive interfaces. Design should correlate with the app's purpose and flow well.

---