// src/data/historyData.js

const CLOUDINARY_BASE_URL = "https://res.cloudinary.com/dfuutwxvq/image/upload/v1781621659";

export const eraData = {
  ancient: {
    title: "DAWN OF SOVEREIGNTY",
    philosophy: "Egypt as the origin of order and the sacred force protecting the Nile Valley.",
    heroes: [
      { id: 1, name: "Seqenenre Tao", img: `${CLOUDINARY_BASE_URL}/seqenenre_nu7rti.jpg`, bio: "The brave Pharaoh who initiated the liberation war against the Hyksos, sacrificing his life for Egypt's sovereignty.", video: "/videos/seqenenre.mp4" },
      { id: 2, name: "Ahmose I", img: `${CLOUDINARY_BASE_URL}/ahmose_a2hwso.jpg`, bio: "The founder of the 18th Dynasty who successfully expelled the Hyksos and reunified the Two Lands.", video: "/videos/ahmose.mp4" },
      { id: 3, name: "Thutmose III", img: `${CLOUDINARY_BASE_URL}/thutmose_m89mbm.jpg`, bio: "A military genius who expanded the Egyptian Empire to its greatest extent, reaching the Euphrates.", video: "/videos/thutmose.mp4" }
    ],
    mapMilestones: [
      { threshold: 0, year: "3500 BC", ruler: "Pre-Dynastic Lords", event: "Formation of Nilotic chiefdoms.", img: `${CLOUDINARY_BASE_URL}/3500_osm0ht.jpg` },
      { threshold: 33, year: "2500 BC", ruler: "Khufu", event: "Peak of Old Kingdom stability.", img: `${CLOUDINARY_BASE_URL}/2500_qynlbt.jpg` },
      { threshold: 66, year: "1500 BC", ruler: "Thutmose III", event: "Empire reaches the Euphrates.", img: `${CLOUDINARY_BASE_URL}/1500_fqhlzt.jpg` },
      { threshold: 100, year: "1000 BC", ruler: "Siamun", event: "Territorial shifts in the Late Period.", img: `${CLOUDINARY_BASE_URL}/1000_rb7lrl.jpg` }
    ]
  },

  greco: {
    title: "BRIDGE OF CIVILIZATIONS",
    philosophy: "Resilience of the Egyptian character against attempts to erase its identity under foreign rule.",
    heroes: [
      { id: 4, name: "Alexander the Great", img: `${CLOUDINARY_BASE_URL}/alexander_thc9dd.jpg`, bio: "The Macedonian conqueror who respected Egyptian traditions and was crowned Pharaoh in Memphis.", video: "/videos/alexander.mp4" },
      { id: 5, name: "Cleopatra VII", img: `${CLOUDINARY_BASE_URL}/Cleopatra_ux5lwx.jpg`, bio: "The last Queen of Egypt, a brilliant polyglot who fought to maintain Egypt's independence from Rome.", video: "/videos/cleopatra.mp4" },
      { id: 6, name: "Pope Benjamin I", img: `${CLOUDINARY_BASE_URL}/Benjamin_pidj15.jpg`, bio: "The 38th Coptic Pope and a symbol of spiritual resistance against Byzantine persecution.", video: "/videos/benjamin.mp4" }
    ],
    mapMilestones: [
      { threshold: 0, year: "500 BC", ruler: "Persian Satraps", event: "Persian occupation of Egypt.", img: `${CLOUDINARY_BASE_URL}/500_nuv4z9.jpg` },
      { threshold: 33, year: "200 BC", ruler: "Ptolemy IV", event: "Ptolemaic naval supremacy.", img: `${CLOUDINARY_BASE_URL}/200_mzmx8d.jpg` },
      { threshold: 66, year: "30 BC", ruler: "Augustus Caesar", event: "Egypt becomes a Roman province.", img: `${CLOUDINARY_BASE_URL}/30_ceweiw.jpg` },
      { threshold: 100, year: "200 AD", ruler: "Septimius Severus", event: "Egypt as Rome's vital breadbasket.", img: `${CLOUDINARY_BASE_URL}/200ad_jwp8w6.jpg` }
    ]
  },

  islamic: {
    title: "FORTRESS OF THE UMMAH",
    philosophy: "Egypt as the heart of the Islamic world and the shield protecting it from annihilation.",
    heroes: [
      { id: 7, name: "Amr ibn al-Aas", img: `${CLOUDINARY_BASE_URL}/amr_dilnbe.jpg`, bio: "The liberator who restored religious freedom and founded Al-Fustat.", video: "/videos/amr.mp4" },
      { id: 8, name: "Salah ad-Din", img: `${CLOUDINARY_BASE_URL}/salaheldin_n6n3vl.jpg`, bio: "The unifier of Egypt and Syria, victor over the Crusaders at Hattin.", video: "/videos/saladin.mp4" },
      { id: 9, name: "Saif ad-Din Qutuz", img: `${CLOUDINARY_BASE_URL}/qutuz_g8m6ba.jpg`, bio: "The hero of Ain Jalut who saved Islamic civilization from the Mongol invasion.", video: "/videos/qutuz.mp4" },
      { id: 10, name: "Baibars", img: `${CLOUDINARY_BASE_URL}/baibars_ydhtmv.jpg`, bio: "The actual founder of the Mamluk Sultanate and defender of the Levant.", video: "/videos/baibars.mp4" }
    ],
    mapMilestones: [
      { threshold: 0, year: "500 AD", ruler: "Byzantine Empire", event: "Egypt before the Arab conquest.", img: `${CLOUDINARY_BASE_URL}/500_l1uyd2.jpg` },
      { threshold: 25, year: "750 AD", ruler: "Abbasid Caliphs", event: "Egypt in the Great Caliphate.", img: `${CLOUDINARY_BASE_URL}/750_j7rdn1.jpg` },
      { threshold: 50, year: "979 AD", ruler: "Al-Mu'izz", event: "Cairo becomes the Fatimid Capital.", img: `${CLOUDINARY_BASE_URL}/979_syocql.jpg` },
      { threshold: 75, year: "1215 AD", ruler: "Al-Adil I", event: "Ayyubid defense against Crusaders.", img: `${CLOUDINARY_BASE_URL}/1215_drtxu8.jpg` },
      { threshold: 100, year: "1453 AD", ruler: "Mamluk Sultans", event: "Peak of Mamluk trade influence.", img: `${CLOUDINARY_BASE_URL}/1453_gqe1jt.jpg` }
    ]
  },

  ottoman: {
    title: "THE PROTECTORATE CALIPHATE",
    philosophy: "A unified Islamic front against European expansion and the preservation of religious identity.",
    heroes: [],
    mapMilestones: [
      { threshold: 0, year: "1648 AD", ruler: "Ottoman Governors", event: "Egypt as a vital Ottoman Eyalet.", img: `${CLOUDINARY_BASE_URL}/1648_ox2yle.jpg` },
      { threshold: 100, year: "1789 AD", ruler: "Murad Bey", event: "The era before the French Campaign.", img: `${CLOUDINARY_BASE_URL}/1789_dyjhnz.jpg` }
    ]
  },

  alawiya: {
    title: "THE IMPERIAL DYNASTY",
    philosophy: "Modernization, architectural elegance, and the vision of a powerful, independent Egyptian Empire.",
    heroes: [
      { id: 11, name: "Muhammad Ali Pasha", img: `${CLOUDINARY_BASE_URL}/m_ali_pu92as.jpg`, bio: "The founder of modern Egypt, the army, and the industrial economy.", video: "/videos/m_ali.mp4" },
      { id: 12, name: "Ibrahim Pasha", img: `${CLOUDINARY_BASE_URL}/ibrahim_ouxeow.jpg`, bio: "The great conqueror who unified the Nile Valley and the Levant.", video: "/videos/ibrahim.mp4" },
      { id: 13, name: "Abbas Helmy II", img: `${CLOUDINARY_BASE_URL}/abbas_yq8frp.jpg`, bio: "The Resistant Khedive who defied British occupation for Egyptian loyalty.", video: "/videos/abbas.mp4" },
      { id: 14, name: "King Farouk I", img: `${CLOUDINARY_BASE_URL}/farouk_xxrs4d.jpg`, bio: "The last King of Egypt, representing a peak of civil elegance and royalty.", video: "/videos/farouk.mp4" }
    ],
    mapMilestones: [
      { threshold: 0, year: "1837 AD", ruler: "Muhammad Ali", event: "The peak of the Egyptian Empire.", img: `${CLOUDINARY_BASE_URL}/1837_ramtnc.jpg` },
      { threshold: 50, year: "1871 AD", ruler: "Khedive Ismail", event: "Expansion into the Horn of Africa.", img: `${CLOUDINARY_BASE_URL}/1871_s1z2fu.jpg` },
      { threshold: 100, year: "1914 AD", ruler: "Hussein Kamel", event: "WWI and the British Protectorate.", img: `${CLOUDINARY_BASE_URL}/1914_dbsyvu.jpg` }
    ]
  },

  modern: {
    title: "REALISM & TRANSFORMATION",
    philosophy: "Analyzing the shift from monarchy to republicanism and its impact on Egyptian identity.",
    heroes: [
      { id: 16, name: "Anwar El-Sadat", img: `${CLOUDINARY_BASE_URL}/sadat_u3donz.jpg`, bio: "The hero of the 1973 war who chose political peace to save the state.", video: "/videos/sadat.mp4" }
    ],
    mapMilestones: [
      { threshold: 0, year: "1960 AD", ruler: "Nasser", event: "High Dam and Post-Revolution borders.", img: `${CLOUDINARY_BASE_URL}/1960_s97dcp.jpg` },
      { threshold: 100, year: "2019 AD", ruler: "Contemporary State", event: "Modern geopolitical borders of Egypt.", img: `${CLOUDINARY_BASE_URL}/2019_ryl03c.jpg` }
    ]
  }
};

export const erasMetadata = [
  { id: 'ancient', label: 'Ancient Egypt', icon: '👑' },
  { id: 'greco', label: 'Greco-Roman', icon: '🏛️' },
  { id: 'islamic', label: 'Islamic Era', icon: '🕌' },
  { id: 'ottoman', label: 'Ottoman Era', icon: '🌙' },
  { id: 'alawiya', label: 'Alawiya Dynasty', icon: '🎩' },
  { id: 'modern', label: 'Modern Egypt', icon: '📉' }
];