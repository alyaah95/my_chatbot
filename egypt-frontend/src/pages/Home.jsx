import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div className="home-content">
     {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <h1>Egyptia 🇪🇬</h1>
          <p>
            Discover the timeless heritage of Egypt. Explore our interactive maps, 
            dive deep into historical eras, and query our intelligent AI assistant to uncover the secrets of the past and present.
          </p>
        </div>
      </section>

      {/* Explore Section */}
      <section className="section">
        <div className="container">
          <h2 className="section-title">Explore Egypt</h2>
          <div className="card-grid">
            {/* Card 1 */}
            <div className="card">
              <img src="https://i.pinimg.com/736x/e0/8e/9f/e08e9f5ad5ba79df900525822fe328d8.jpg" alt="Pyramids" className="card-img" />
              <div className="card-content">
                <h3>History</h3>
                <p>Discover the fascinating history of ancient Egypt, from the early dynastic period to the Roman conquest.</p>
                <Link to="/history"> <button className="btn">Explore</button> </Link>
                
              </div>
            </div>
            {/* Card 2 */}
            <div className="card">
              <img src="https://i.pinimg.com/736x/43/d9/98/43d998b8b7812b23c858f6572a9e8698.jpg" alt="Nile" className="card-img" />
              <div className="card-content">
                <h3>Geography</h3>
                <p>Explore Egypt's diverse geography, from the Nile Valley to the deserts and coastal regions.</p>
                <Link to="/geography"> <button className="btn">Explore</button> </Link>
                
              </div>
            </div>
            {/* Card 3 */}
            <div className="card">
              <img src="https://i.pinimg.com/1200x/79/56/88/795688789a0643011bddf2f1e7426eb7.jpg" alt="AI" className="card-img" />
              <div className="card-content">
                <h3>AI Assistant</h3>
                <p>Ask our AI assistant any questions about Egypt and get detailed, accurate answers.</p>
                <Link to="/assistant"> <button className="btn">Try Now</button> </Link> 
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Did You Know Section */}
      <section className="section">
        <div className="container">
          <h2 className="section-title">Did You Know?</h2>
          <div className="card-grid">
            <div className="card"><div className="card-content"><h3>Ancient Inventions</h3><p>Paper, toothpaste, and the 365-day calendar.</p></div></div>
            <div className="card"><div className="card-content"><h3>Longest River</h3><p>The Nile is the longest river in the world (6,650 km).</p></div></div>
            <div className="card"><div className="card-content"><h3>Pyramid Precision</h3><p>The Great Pyramid was the tallest structure for 3,800 years.</p></div></div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;