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


// const hotspots = [
//   { 
//     id: 'tiran', 
//     name: 'Tiran & Sanafir', 
//     x: 413,
//     y: 208,
//     info: 'Tiran and Sanafir are two strategic Egyptian islands located at the entrance of the Gulf of Aqaba.' ,
//     image: 'https://i.pinimg.com/736x/73/db/a7/73dba70a42776116c45709e4d46c99cf.jpg',
//     gallery: ["https://i.pinimg.com/736x/60/10/8a/60108a84d46c0c37426c33953aaa5231.jpg",
//       "https://media.almalnews.com/2015/3/large/b737149e-6408-4dac-a053-617249f47d8f.jpg",
//     ]
//   },
//   { 
//     id: 'halaib', 
//     name: 'Halaib & Shalatin', 
//     x: 490, 
//     y: 460, 
//     info: 'The southeastern crown of Egypt, Halaib and Shalatin are home to the spectacular Gabal Elba National Park. This Egyptian territory is a sanctuary of unique biodiversity, ancient tribal heritage, and untouched Red Sea beauty.',
//     image:"https://i.pinimg.com/736x/05/15/cf/0515cf8ee9711f5899d7e7431e3c06ea.jpg",
//     gallery:["https://i.pinimg.com/736x/38/8f/9c/388f9c11772c116b11851f90da53fee5.jpg",
//       "https://i.pinimg.com/736x/4c/f3/00/4cf300d91a9db4c5c742db90b08ba442.jpg",
//       "https://i.pinimg.com/736x/cd/b4/d8/cdb4d8e7e15205e1624d4c3a2e02fa60.jpg"]
//   },
//   { 
//     id: 'Rafah', 
//     name: 'Rafah', 
//     x: 428, 
//     y: 29, 
//     info: 'Egypt\'s border with the Gaza Strip.',
//     image: 'https://vid.alarabiya.net/images/2025/01/31/675cebed-5157-43b1-9efd-a73be821b0db/675cebed-5157-43b1-9efd-a73be821b0db_16x9_1200x676.jpg',
//     gallery: ['https://img.youm7.com/large/2024042402030535.jpg',
//       'https://i.pinimg.com/1200x/7e/b5/b3/7eb5b3398031b13038169f52c198b44f.jpg',
//       'https://images.skynewsarabia.com/images/v1/2024/04/25/1708994/800/450/1-1708994.JPG',
//       'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRrz8trW33GmzOU8JS8ZE8kubcTqyo35bQOaw&s'
//     ]
//   },
  
// ];

// const natureWonders = [
//   {
//     id: 'Siwa',
//     title: 'Siwa Oasis',
//     subtitle: 'The Pearl of the Western Desert',
//     mainImage: defaultImg,
//     gallery: [defaultImg, defaultImg , defaultImg , defaultImg , defaultImg],
//     description: 'Siwa Oasis is a unique blend of natural beauty and cultural heritage, known for its lush palm groves, salt lakes, and ancient ruins.'
//   },
//   {
//     id: 'Nile',
//     title: 'The Eternal Nile',
//     subtitle: 'The Lifeblood of Egypt',
//     mainImage: '/assets/nature/default.jpg',
//     gallery: ['/assets/nature/default.jpg'],
//     description: 'The Nile River is not just a waterway; it is the lifeblood that has shaped Egyptian civilization through the ages.'
//   },
//   // أضيفي باقي الأماكن هنا بنفس الطريقة
//   {
//     id: 'St-Catherine',
//     title: 'St. Catherine',
//     subtitle: 'The Sacred Peaks of Sinai',
//     mainImage: defaultImg,
//     gallery: [defaultImg, defaultImg, defaultImg],
//     description: 'Home to the highest peaks in Egypt, St. Catherine is a mystical landscape of rugged granite mountains and ancient monastic heritage, offering breathtaking sunrise views.'
// },
// {
//     id: 'White-Desert',
//     title: 'White Desert',
//     subtitle: 'A Lunar Landscape on Earth',
//     mainImage: defaultImg,
//     gallery: [defaultImg, defaultImg, defaultImg],
//     description: 'The Farafra depression hosts alien-looking chalk rock formations that turn the desert into a surreal white wonderland, sculpted by centuries of wind and sand.'
// },
// {
//     id: 'Ras-Mohammed',
//     title: 'Ras Mohammed',
//     subtitle: 'Where Two Gulfs Meet',
//     mainImage: defaultImg,
//     gallery: [defaultImg, defaultImg, defaultImg],
//     description: 'Located at the southernmost tip of Sinai, this world-renowned marine park features vibrant coral reefs, mangrove forests, and a diverse array of marine life.'
// },
// {
//     id: 'Whale-Valley',
//     title: 'Wadi Al-Hitan',
//     subtitle: 'The Valley of Whales',
//     mainImage: defaultImg,
//     gallery: [defaultImg, defaultImg, defaultImg],
//     description: 'A UNESCO World Heritage site that holds hundreds of fossils of some of the earliest forms of whales, telling a 40-million-year-old story of evolutionary history.'
// },

// ];

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

                {/* النص الشمال: الكلام */}
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
            {/* stopPropagation عشان الكارت ميقفلش لما ندوس جواه */}
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