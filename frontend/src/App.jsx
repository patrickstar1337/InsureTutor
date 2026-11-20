import { useEffect, useRef, useState } from 'react'

const examples = [
  'What is the guaranteed interest rate?',
  'Could I receive less if I surrender the policy early?',
  '冷靜期內如何取消保單？',
]

function SourceList({ sources }) {
  if (!sources?.length) return null

  return (
    <div className="sources">
      <p className="sources-heading">Retrieved passages · not verified citations</p>
      <ul>
        {sources.map((source) => (
          <li key={source.chunk_id}>
            {source.source} · PDF {source.pdf_pages.length === 1 ? 'page' : 'pages'}{' '}
            {source.pdf_pages.join(', ')}
          </li>
        ))}
      </ul>
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [pendingQuestion, setPendingQuestion] = useState('')
  const [error, setError] = useState('')
  const chatLogRef = useRef(null)
  const busy = Boolean(pendingQuestion)

  useEffect(() => {
    if (chatLogRef.current) {
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight
    }
  }, [messages, pendingQuestion, error])

  async function sendQuestion(event) {
    event.preventDefault()
    const question = draft.trim()
    if (!question || busy) return

    setError('')
    setPendingQuestion(question)
    setDraft('')

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          history: messages.slice(-4).map(({ role, content }) => ({ role, content })),
        }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || 'The answer service is unavailable.')
      }
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', content: question },
        { id: crypto.randomUUID(), role: 'assistant', content: data.answer, sources: data.sources },
      ])
    } catch (requestError) {
      setDraft(question)
      setError(
        requestError instanceof TypeError
          ? 'Cannot reach the backend. Check that FastAPI is running.'
          : requestError.message,
      )
    } finally {
      setPendingQuestion('')
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true">I</div>
        <div>
          <p className="brand-name">InsureTutor</p>
          <p className="brand-caption">Understand the brochure, one question at a time</p>
        </div>
      </header>

      <main className="main-content">
        <div className="intro">
          <span className="eyebrow">Insurance document guide</span>
          <h1>Ask about the policy brochure.</h1>
          <p>Get a plain-language answer based on the supplied FLEXI-ULife Prime Saver brochure.</p>
        </div>

        <section className="chat-card" aria-label="InsureTutor chat">
          <div className="chat-log" ref={chatLogRef}>
            {messages.length === 0 && !pendingQuestion && (
              <div className="empty-state">
                <h2>What would you like to understand?</h2>
                <p>You can ask in English or Chinese. Try one of these questions:</p>
                <div className="examples">
                  {examples.map((example) => (
                    <button key={example} type="button" onClick={() => setDraft(example)}>
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div className={`message-row ${message.role}`} key={message.id}>
                <article className="message-bubble">
                  <p className="message-label">{message.role === 'user' ? 'You' : 'InsureTutor'}</p>
                  <p className="message-text">{message.content}</p>
                  {message.role === 'assistant' && <SourceList sources={message.sources} />}
                </article>
              </div>
            ))}

            {pendingQuestion && (
              <>
                <div className="message-row user">
                  <article className="message-bubble">
                    <p className="message-label">You</p>
                    <p className="message-text">{pendingQuestion}</p>
                  </article>
                </div>
                <div className="message-row assistant" role="status">
                  <article className="message-bubble loading">Finding relevant passages and writing an answer…</article>
                </div>
              </>
            )}
          </div>

          <div className="composer-area">
            {error && <p className="error-message" role="alert">{error}</p>}
            <form onSubmit={sendQuestion} className="composer">
              <label htmlFor="question" className="visually-hidden">Your question</label>
              <textarea
                id="question"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about the brochure…"
                rows={2}
                disabled={busy}
              />
              <button type="submit" disabled={busy || !draft.trim()}>
                {busy ? 'Working…' : 'Send'}
              </button>
            </form>
            <p className="composer-note">Follow-up questions use recent chat for context. Check the policy document for full terms.</p>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
