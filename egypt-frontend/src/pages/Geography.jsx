import React, { useState } from 'react';
import EgyptMap from '../components/EgyptMap';
import './Geography.css';
import defaultImg from '../assets/default.jpg';
import { hotspots, natureWonders } from '../data/hotspotsAndWondersData';

import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Pagination, Autoplay } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/pagination';



function Geography() {
  const [selectedSpot, setSelectedSpot] = useState(null);
  const [selectedHotspot, setSelectedHotspot] = useState(null);
  const [activeNature, setActiveNature] = useState(null);
  const [showFullGallery, setShowFullGallery] = useState(false);

const handleProvinceClick = (data) => {
    setSelectedHotspot(null);
    setActiveNature(null);
    setShowFullGallery(false);
    setSelectedSpot({
      name: data.name,
      info: data.info,
      img: data.img,
      gallery: data.gallery
    });
  };
  const handleHotspotClick = (spot) => {
    setSelectedSpot(null);
    setActiveNature(null);
    setShowFullGallery(false);
    setSelectedHotspot(spot);
  };

  return (
    <div className="geography-container">
      {/* 1. Header Area - Glassmorphism */}
      <section className="geo-intro-section">
        <h2 className="section-title">GEOGRAPHIC SOVEREIGNTY</h2>
        <div className="gold-line-centered"></div>
        <p className="section-subtitle">Explore the sacred borders and natural wonders of Egypt</p>
      </section>

      {/* 2. THE LIVE MAP SECTION */}
      <section className="map-hero-section">
        <div className="map-wrapper">
          <EgyptMap onProvinceClick={handleProvinceClick} hotspots={hotspots} onHotspotClick={handleHotspotClick}/>
          {selectedHotspot && (
            <>
            <div className="info-modal-overlay" onClick={() => setSelectedSpot(null)}>
              <div className="modern-modal" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close" onClick={() => {
                    setSelectedHotspot(null);
                    setShowFullGallery(false);
                }}>×</button>
                
                <div className="modal-image-side">
                  <img 
                    src={selectedHotspot.image || "src/assets/nature/default.jpg"} 
                    alt={selectedHotspot.name} 
                  />
                </div>


                <div className="modal-content-side">
                  <h2>{selectedHotspot.name}</h2>
                  <div className="modal-divider"></div>
                  <p>{selectedHotspot.info}</p>
                  <button 
                  className={`explore-more-btn ${showFullGallery ? 'active' : ''}`}
                  onClick={() => setShowFullGallery(true)} // نفتح الجاليري
                >
                  Explore Full Gallery
                </button>
                </div>
                {showFullGallery && (
                  <div className="full-gallery-overlay-slider"> 
                    <button className="close-gallery-btn" onClick={() => setShowFullGallery(false)}>
                      BACK TO INFO ×
                    </button>
                    
                    <Swiper
                      modules={[Navigation, Pagination]}
                      navigation={true}
                      pagination={{ clickable: true }}
                      className="full-swiper"
                    >
                      {selectedHotspot.gallery?.map((img, index) => (
                        <SwiperSlide key={index}>
                          <img src={img || defaultImg} className="full-view-img" alt="Gallery View" />
                        </SwiperSlide>
                      ))}
                    </Swiper>
                  </div>
                )}
              </div>
            </div>
            </>
          )}
        </div>


        {selectedSpot && (
          <>
          <div className="info-modal-overlay" onClick={() => setSelectedSpot(null)}>

            <div className="modern-modal" onClick={(e) => e.stopPropagation()}>
              <button className="modal-close" onClick={() => {
                  setSelectedSpot(null);
                  setShowFullGallery(false);
              }}>×</button>
              
              <div className="modal-image-side">
                <img 
                  src={selectedSpot.img || "src/assets/nature/default.jpg"} 
                  alt={selectedSpot.name} 
                />
              </div>

              <div className="modal-content-side">
                <h2>{selectedSpot.name}</h2>
                <div className="modal-divider"></div>
                <p>{selectedSpot.info}</p>
                <button 
                  className={`explore-more-btn ${showFullGallery ? 'active' : ''}`}
                  onClick={() => setShowFullGallery(true)}
                >
                  Explore Full Gallery
                </button>
                
              </div>
              {showFullGallery && (
                <div className="full-gallery-overlay-slider"> 
                  <button className="close-gallery-btn" onClick={() => setShowFullGallery(false)}>
                    BACK TO INFO ×
                  </button>
                  
                  <Swiper
                    modules={[Navigation, Pagination]}
                    navigation={true}
                    pagination={{ clickable: true }}
                    className="full-swiper"
                  >
                    {selectedSpot.gallery && selectedSpot.gallery.map((img, index) => (
                      <SwiperSlide key={index}>
                        <img 
                          src={img || defaultImg} 
                          className="full-view-img" 
                          alt={`Sinai View ${index}`} 
                          onError={(e) => e.target.src = defaultImg}
                        />
                      </SwiperSlide>
                    ))}
                  </Swiper>
                </div>
              )}             
            </div> 
          </div>
          
          </>
        )}
      </section>
      {/* 3. NATURAL GALLERY SECTION */}
      <section className="nature-grid-section">
        <h2 className="section-title">NATURAL WONDERS</h2>
        <div className="gold-line-centered"></div>
        
        <div className="nature-slider-outer-container">
          <Swiper
            modules={[Navigation, Pagination]}
            spaceBetween={30}
            slidesPerView={1}
            navigation={true} 
            pagination={{ clickable: true, dynamicBullets: true }}
            breakpoints={{
              768: { slidesPerView: 2 },
              1024: { slidesPerView: 3 }
            }}
            className="main-nature-swiper"
          >
            {natureWonders.map((place) => (
              <SwiperSlide key={place.id}> 
                <div className="nature-card" onClick={() => {
                  setActiveNature(place);
                  setShowFullGallery(false);
                }}>
                  <img src={place.mainImage} alt={place.title} />
                  <div className="nature-card-overlay">
                    <span>{place.title}</span>
                    <p>Explore Discovery</p>
                  </div>
                </div>
              </SwiperSlide>
            ))}
          </Swiper>
        </div>
      </section>

{activeNature && (
  <div className="info-modal-overlay" onClick={() => setActiveNature(null)}>
    <div className="modern-modal" onClick={(e) => e.stopPropagation()}>
      <button className="modal-close" onClick={() => setActiveNature(null)}>×</button>
      
      <div className="modal-image-side">
         
           <img src={activeNature.mainImage} alt={activeNature.title} className="main-modal-img" />
         
      </div>

      <div className="modal-content-side">
        <span className="location-tag">Egypt / Nature</span>
        <h2>{activeNature.title}</h2>
        <div className="modal-divider"></div>
        <p>{activeNature.description}</p>
        
        <button 
          className={`explore-more-btn ${showFullGallery ? 'active' : ''}`}
          onClick={() => setShowFullGallery(true)}
        >
          Explore Full Gallery
        </button>
      </div>
      {showFullGallery && (
        <div className="full-gallery-overlay-slider">
          <button className="close-gallery-btn" onClick={() => setShowFullGallery(false)}>
            Back to Info ←
          </button>
          
          <Swiper
            modules={[Navigation, Pagination]}
            navigation={true}
            pagination={{ type: 'fraction' }}
            className="full-swiper"
          >
            {activeNature.gallery.map((img, idx) => (
              <SwiperSlide key={`${activeNature.id}-full-${idx}`}>
                <img src={img} alt="Nature Full View" className="full-view-img" />
              </SwiperSlide>
            ))}
          </Swiper>
        </div>
      )}
    </div>
  </div>
)}
    </div>
  );
}

export default Geography;