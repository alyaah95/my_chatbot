import React, { useState } from 'react';
import './History.css';


import { eraData, erasMetadata } from '../data/historyData';

function History() {
  const [selectedEra, setSelectedEra] = useState('ancient');
  const [selectedHero, setSelectedHero] = useState(null);
  const [mapScale, setMapScale] = useState(0);

  const currentData = eraData[selectedEra] || { title: selectedEra.toUpperCase(), heroes: [], philosophy: "", mapMilestones: [] };


  const currentMilestone = currentData.mapMilestones && currentData.mapMilestones.length > 0
    ? [...currentData.mapMilestones].reverse().find(m => mapScale >= m.threshold) || currentData.mapMilestones[0]
    : null;

  return (
    <div className="history-page-layout">
      
      {/* 1. LEFT SIDEBAR: Timeline */}
      <aside className="timeline-sidebar">
        <div className="timeline-line"></div>
        {erasMetadata.map((era) => (
          <div 
            key={era.id} 
            className={`timeline-dot-wrapper ${selectedEra === era.id ? 'active' : ''}`}
            onClick={() => {
              setSelectedEra(era.id);
              setSelectedHero(null);
              setMapScale(0);
            }}
          >
            <div className="timeline-dot">{era.icon}</div>
            <span className="era-label-sidebar">{era.label}</span>
          </div>
        ))}
      </aside>

      {/* 2. Main Content Area */}
      <main className="history-main-content">
        
        {/* TOP: Image Gallery (Network of Portraits) */}
        <section className="heroes-section">
          <h2 className="era-title-display">{currentData.title}</h2>
          <div className="heroes-gallery">
            {currentData.heroes.map((hero, index) => (
              <div 
                key={hero.id} 
                className={`portrait-card ${index % 2 === 0 ? 'card-even' : 'card-odd'}`}
                onClick={() => setSelectedHero(hero)}
              >
                <div className="portrait-frame">
                  <img src={hero.img} alt={hero.name} />
                  <div className="portrait-overlay">
                    <button className="explore-btn">Discover Story</button> 
                  </div>
                </div>
                <h3>{hero.name}</h3>
              </div>
            ))}
          </div>
        </section>

        <hr className="section-divider" />

        {/* BOTTOM: Era Summary & Map */}
        <section className="era-details-section">
          <div className="summary-box-full">
            <h3>Era Philosophy</h3>
            <div className="gold-line-short"></div>
            <p>{currentData.philosophy}</p>
          </div>

          {/* 2. Interactive Map Section */}
          <section className="map-expansion-container">
            <div className="map-header">
              <h3>Geographic Expansion</h3>
              <p>Slide through the timeline to witness the shifting borders</p>
            </div>


            <div className="map-info-overlay">
              {currentMilestone && (
                <div className="milestone-info fade-in" key={currentMilestone.year}>
                  <span className="milestone-year">{currentMilestone.year}</span>
                  <span className="milestone-ruler">Under: <strong>{currentMilestone.ruler}</strong></span>
                  <p className="milestone-event">{currentMilestone.event}</p>
                </div>
              )}
            </div>

            <div className="map-visual-wrapper">
              {currentMilestone && (
                <img 
                  src={currentMilestone.img} 
                  className="map-layer fade-in" 
                  alt={currentMilestone.year}
                  key={currentMilestone.img} 
                />
              )}
            </div>

            <div className="map-timeline-control">
              <span>Start of Era</span>
              <input 
                type="range" 
                min="0" max="100" 
                value={mapScale} 
                onChange={(e) => setMapScale(e.target.value)} 
                className="map-slider"
              />
              <span>Peak Power</span>
            </div>
          </section>
        </section>

        {/* 3. CENTER MODAL (The improved Hero Details) */}
        {selectedHero && (
          <div className="hero-modal-overlay" onClick={() => setSelectedHero(null)}>
            <div className="hero-modal-content" onClick={e => e.stopPropagation()}>
              <button className="close-modal" onClick={() => setSelectedHero(null)}>×</button>

              <div className="modal-main-container">
                {/* 1. Large Video Section (60%) */}
                <div className="modal-video-box">
                  <video controls width="100%" autoPlay>
                    <source src={selectedHero.video} type="video/mp4" />
                  </video>
                </div>

                {/* 2. Focused Info Section (40%) */}
                <div className="modal-info-box">
                  <span className="modal-subtitle">HISTORICAL FIGURE</span>
                  <h2>{selectedHero.name}</h2>
                  <div className="divider-gold"></div>
                  <div className="bio-scroll-area">
                    <p>{selectedHero.bio}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default History;