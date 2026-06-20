import React, { useState } from 'react';

function Assistant() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('Your answer will appear here...');
  const [loading, setLoading] = useState(false);
  const [fullAnswer, setFullAnswer] = useState('');
  const [showSummarize, setShowSummarize] = useState(false);
  const [showFullBtn, setShowFullBtn] = useState(false);

  const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
  const ASK_API_URL = `${BASE_URL}/ask`;
  const SUMMARIZE_API_URL = `${BASE_URL}/summarize`;

  const handleAsk = async () => {
    if (!question.trim()) {
      setAnswer("Please enter a question.");
      return;
    }

    setLoading(true);
    setAnswer("Searching for an answer...");
    setShowSummarize(false);
    setShowFullBtn(false);

    try {
      const response = await fetch(ASK_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question })
      });

      const data = await response.json();
      if (response.ok) {
        setAnswer(data.answer);
        setFullAnswer(data.answer);
        setShowSummarize(true);
      } else {
        throw new Error(data.answer || 'Server error');
      }
    } catch (error) {
      setAnswer(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSummarize = async () => {
    setAnswer("Summarizing...");
    try {
      const response = await fetch(SUMMARIZE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: fullAnswer, question: question })
      });
      const data = await response.json();
      setAnswer(data.summary);
      setShowSummarize(false);
      setShowFullBtn(true);
    } catch (error) {
      setAnswer("Error during summarization.");
    }
  };

  return (
    <div className="assistant-page">
      <section className="hero">
        <div className="container">
          <h1>Egypt AI Assistant</h1>
          <p>Ask questions about Egypt's history and geography</p>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="assistant-container">
            <h2 className="section-title">Ask About Egypt</h2>
            <div className="input-group">
              <input 
                type="text" 
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
                placeholder="Enter your question about Egypt..." 
              />
              <button onClick={handleAsk} className="btn" disabled={loading}>
                {loading ? '...' : 'Ask'}
              </button>
            </div>

            <div className="answer-area">
              <p>{answer}</p>
            </div>

            <div className="button-group">
              {showSummarize && <button onClick={handleSummarize} className="btn btn-secondary">Summarize Answer</button>}
              {showFullBtn && <button onClick={() => { setAnswer(fullAnswer); setShowFullBtn(false); setShowSummarize(true); }} className="btn btn-secondary">Show Full Answer</button>}
            </div>
          </div>

          <div className="card-grid">
            <div className="card">
              <div className="card-content">
                <h3>Sample Questions</h3>
                <p>• What was the main characteristic of the New Kingdom?</p>
                <p>• Tell me about the Suez Canal significance.</p>
              </div>
            </div>
            <div className="card">
              <div className="card-content">
                <h3>About Our Assistant</h3>
                <p>Our AI uses advanced RAG models to provide accurate info from our curated knowledge base.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Assistant;