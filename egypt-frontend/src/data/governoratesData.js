

const BASE    = "https://res.cloudinary.com/dfuutwxvq/image/upload";
const TR      = "w_600,h_400,c_pad,b_auto";
const VER     = "v1781621685";
const url     = (name) => `${BASE}/${TR}/${VER}/${name}.jpg`;

//helper: build a gallery array from raw Cloudinary names
const gallery = (...names) => names.map(url);

export const placesData = [
  // ── Alexandria ──────────────────────────────────────────────────────────
  {
    id: 1,
    govId: "EG-ALX",
    name: "Alexandria",
    category: "Governorates",
    info: "A historic Mediterranean port city known for its rich cultural heritage and iconic landmarks.",
    img: url("alex1_fsp9up"),
    gallery: gallery("alex2_cc5lvy", "alex3_jwjjc0", "alex4_jdkfel", "alex5_sd7kex"),
  },
  // ── Aswan ────────────────────────────────────────────────────────────────
  {
    id: 2,
    govId: "EG-ASN",
    name: "Aswan",
    category: "Governorates",
    info: "A serene city on the Nile, known for Nubian culture, the High Dam, and Philae Temple.",
    img: url("aswan1_gawwvg"),
    gallery: gallery("aswan2_vh62h0", "aswan3_giubwr", "aswan4_cqn3fb"),
  },
  // ── Asyut ────────────────────────────────────────────────────────────────
  {
    id: 3,
    govId: "EG-AST",
    name: "Asyut",
    category: "Governorates",
    info: "The cultural and commercial heart of Upper Egypt, home to a major university and monasteries.",
    img: url("asyut1_vfgtp6"),
    gallery: gallery("asyut2_ptzg2d", "asyut3_zhkauz", "asyut4_mybtjp"),
  },
  // ── Red Sea ──────────────────────────────────────────────────────────────
  {
    id: 4,
    govId: "EG-BA",
    name: "Red Sea",
    category: "Governorates",
    info: "A premier tourist destination known for Hurghada, crystal-clear waters, and coral reefs.",
    img: url("red_sea1_witfen"),
    gallery: gallery(
      "red_sea2_vbnzfk", "red_sea3_emri6f", "red_sea4_pk0ztr",
      "red_sea5_wr9jfo", "red_sea6_yiom6b", "red_sea7_k9tkkv"
    ),
  },
  // ── Beheira ──────────────────────────────────────────────────────────────
  {
    id: 5,
    govId: "EG-BH",
    name: "Beheira",
    category: "Governorates",
    info: "A major agricultural hub in the Delta, home to the historic city of Rosetta (Rashid).",
    img: url("beheira1_xeutnj"),
    gallery: gallery("beheira2_k3nhbu", "beheira4_vq5abu"),
  },
  // ── Beni Suef ────────────────────────────────────────────────────────────
  {
    id: 6,
    govId: "EG-BNS",
    name: "Beni Suef",
    category: "Governorates",
    info: "A bridge between Lower and Upper Egypt, home to the unique Meidum Pyramid.",
    img: url("beni_suef1_boqdyd"),
    gallery: gallery("beni_suef2_xfvorg", "beni_suef3_him9vg"),
  },
  // ── Cairo ────────────────────────────────────────────────────────────────
  {
    id: 7,
    govId: "EG-C",
    name: "Cairo",
    category: "Governorates",
    info: "The vibrant capital of Egypt, home to historic Islamic architecture and the majestic Nile.",
    img: url("cairo1_ijll8z"),
    gallery: gallery("cairo2_atppwh", "cairo3_xfrcex", "cairo4_aiibsl", "cairo5_wtmgiz"),
  },
  // ── Dakahlia ─────────────────────────────────────────────────────────────
  {
    id: 8,
    govId: "EG-DK",
    name: "Dakahlia",
    category: "Governorates",
    info: "Known for its capital Mansoura, a city with a rich history and cultural significance in the Delta.",
    img: url("dakahlia1_jraajs"),
    gallery: gallery("dakahlia2_y9co8m", "dakahlia3_agfeou"),
  },
  // ── Damietta ─────────────────────────────────────────────────────────────
  {
    id: 9,
    govId: "EG-DT",
    name: "Damietta",
    category: "Governorates",
    info: "A coastal city famous for its furniture industry and the Ras El Bar resort.",
    img: url("damietta2_wj6w1b"),
    gallery: gallery("damietta2_wj6w1b"),
  },
  // ── Fayoum ───────────────────────────────────────────────────────────────
  {
    id: 10,
    govId: "EG-FYM",
    name: "Fayoum",
    category: "Governorates",
    info: "An ancient oasis known for the Qarun Lake and the UNESCO site Wadi Al-Hitan.",
    img: url("fayoum2_zqdfgm"),
    gallery: gallery("fayoum2_zqdfgm", "fayoum3_wrcetd"),
  },
  // ── Gharbia ──────────────────────────────────────────────────────────────
  {
    id: 11,
    govId: "EG-GH",
    name: "Gharbia",
    category: "Governorates",
    info: "Located in the heart of the Delta, famous for Tanta and the historic El-Sayed El-Badawi Mosque.",
    img: url("gharbia1_oobcdd"),
    gallery: gallery("gharbia2_sokkoo", "gharbia3_dladkm"),
  },
  // ── Giza ─────────────────────────────────────────────────────────────────
  {
    id: 12,
    govId: "EG-GZ",
    name: "Giza",
    category: "Governorates",
    info: "World-famous for the Great Pyramids and the Sphinx, representing the heart of Ancient Egypt.",
    img: url("giza1_e4rtbt"),
    gallery: gallery("giza2_qjrmts"),
  },
  // ── Ismailia ─────────────────────────────────────────────────────────────
  {
    id: 13,
    govId: "EG-IS",
    name: "Ismailia",
    category: "Governorates",
    info: "Known as the City of Gardens, it is the headquarters of the Suez Canal Authority.",
    img: url("ismailia1_onsxiy"),
    gallery: gallery("ismailia3_ibvv8k", "ismailia4_s0s9tz"),
  },
  // ── South Sinai ──────────────────────────────────────────────────────────
  {
    id: 14,
    govId: "EG-JS",
    name: "South Sinai",
    category: "Governorates",
    info: "Home to world-class diving resorts like Sharm El-Sheikh and the sacred Mount Catherine.",
    img: url("south_sinai1_qekhpp"),
    gallery: gallery(
      "south_sinai2_teshpb", "south_sinai3_j7tlmk",
      "south_sinai4_wldsnu", "south_sinai5_tbbb7d"
    ),
  },
  // ── Qalyubia ─────────────────────────────────────────────────────────────
  {
    id: 15,
    govId: "EG-KB",
    name: "Qalyubia",
    category: "Governorates",
    info: "Part of Greater Cairo, famous for its fruit orchards and the Barrage (El-Kanater).",
    img: url("qalyubia1_oil8ug"),
    gallery: gallery("qalyubia2_vx4vdm", "qalyubia3_jjhm4u"),
  },
  // ── Kafr El Sheikh ───────────────────────────────────────────────────────
  {
    id: 16,
    govId: "EG-KFS",
    name: "Kafr El Sheikh",
    category: "Governorates",
    info: "A beautiful Delta region known for Lake Burullus and its rich fishing industry.",
    img: url("kafr_el_sheikh1_lxpaqd"),
    gallery: gallery(
      "kafr_el_sheikh2_z5z5fr", "kafr_el_sheikh3_ru9par", "kafr_el_sheikh4_vqrmvc"
    ),
  },
  // ── Qena ─────────────────────────────────────────────────────────────────
  {
    id: 17,
    govId: "EG-KN",
    name: "Qena",
    category: "Governorates",
    info: "Known for the magnificent Dendera Temple complex, one of Egypt's best-preserved sites.",
    img: url("qena1_ac6knt"),
    gallery: gallery("qena2_ea2dqi"),
  },
  // ── Luxor ────────────────────────────────────────────────────────────────
  {
    id: 18,
    govId: "EG-LX",
    name: "Luxor",
    category: "Governorates",
    info: "Often called the world's greatest open-air museum, containing a third of the world's monuments.",
    img: url("red_sea1_witfen"), // placeholder – add luxor images to Cloudinary
    gallery: [],
  },
  // ── Minya ────────────────────────────────────────────────────────────────
  {
    id: 19,
    govId: "EG-MN",
    name: "Minya",
    category: "Governorates",
    info: "Known as the Bride of Upper Egypt, it holds vast archaeological treasures like Beni Hasan.",
    img: url("minya1_iyjnhc"),
    gallery: gallery("minya2_ilvv9g", "minya3_gqp3rv", "minya4_unwvgq"),
  },
  // ── Monufia ──────────────────────────────────────────────────────────────
  {
    id: 20,
    govId: "EG-MNF",
    name: "Monufia",
    category: "Governorates",
    info: "A central Delta governorate known for its fertile lands and being the birthplace of many leaders.",
    img: url("monufia1_deihpw"),
    gallery: gallery(
      "monufia2_zrot3n", "monufia3_val1dm", "monufia4_jy0ieb", "monufia5_yqfbtt"
    ),
  },
  // ── Matrouh ──────────────────────────────────────────────────────────────
  {
    id: 21,
    govId: "EG-MT",
    name: "Matrouh",
    category: "Governorates",
    info: "Home to the turquoise waters of Marsa Matrouh and the enchanting Siwa Oasis.",
    img: url("matrouh1_adzzfq"),
    gallery: gallery(
      "matrouh2_tckznu", "matrouh3_klitux", "matrouh4_zegrj4", "matrouh5_pghc7z"
    ),
  },
  // ── Port Said ────────────────────────────────────────────────────────────
  {
    id: 22,
    govId: "EG-PTS",
    name: "Port Said",
    category: "Governorates",
    info: "A strategic port city at the northern entrance of the Suez Canal with unique architectural charm.",
    img: url("port_said1_lpobsv"),
    gallery: gallery(
      "port_said2_j9vszs", "port_said3_ocb1gc",
      "port_said4_fz6nvz", "port_said5_l4eblt"
    ),
  },
  // ── Sohag ────────────────────────────────────────────────────────────────
  {
    id: 23,
    govId: "EG-SHG",
    name: "Sohag",
    category: "Governorates",
    info: "Rich in religious history, home to the White and Red Monasteries and Abydos Temple.",
    img: url("sohag1_isjd0t"),
    gallery: gallery("sohag2_tct16v", "sohag3_lvizqp"),
  },
  // ── Sharqia ──────────────────────────────────────────────────────────────
  {
    id: 24,
    govId: "EG-SHR",
    name: "Sharqia",
    category: "Governorates",
    info: "Famous for horse breeding and the ancient archaeological site of Tell Basta.",
    img: url("sharqia1_jncdby"),
    gallery: gallery("sharqia2_sfj4vu", "sharqia3_kvqaah", "sharqia4_zt5mwp"),
  },
  // ── North Sinai ──────────────────────────────────────────────────────────
  {
    id: 25,
    govId: "EG-SIN",
    name: "North Sinai",
    category: "Governorates",
    info: "Famous for its stunning Mediterranean coastline and the Bardawil Lake.",
    img: url("north_sinai1_zxxmf6"),
    gallery: gallery(
      "north_sinai2_s9bwdr", "north_sinai3_eziq7c",
      "north_sinai4_we50fg", "north_sinai5_awrrql"
    ),
  },
  // ── Bir Tawil ────────────────────────────────────────────────────────────
  {
    id: 26,
    govId: "EG-TER",
    name: "Bir Tawil area",
    category: "Governorates",
    info: "Land without an owner — a unique 2,000 km² trapezoid between Egypt and Sudan.",
    img: url("bir_tawil_area1_m02umk"),
    gallery: gallery("bir_tawil_area2_fl1x9j"),
  },
  // ── Halaib & Shalatin ────────────────────────────────────────────────────
  {
    id: 27,
    govId: "EG-HT",
    name: "Halaib & Shalatin",
    category: "Governorates",
    info: "Egypt's tropical gateway, famous for El Elba Protected Area and stunning Red Sea beaches.",
    img: url("halaib_and_shalatin1_vzdrsf"),
    gallery: gallery(
      "halaib_and_shalatin2_bjmiwi",
      "halaib_and_shalatin3_syni47",
      "halaib_and_shalatin4_er9nnb"
    ),
  },
  // ── Suez ─────────────────────────────────────────────────────────────────
  {
    id: 28,
    govId: "EG-SUZ",
    name: "Suez",
    category: "Governorates",
    info: "A historic city on the Red Sea and a vital hub for global maritime trade through the Suez Canal.",
    img: url("suez1_uqajtx"),
    gallery: gallery("suez2_gekgv4", "suez3_uvbflw", "suez4_trarot", "suez5_iicfby"),
  },
  // ── New Valley ───────────────────────────────────────────────────────────
  {
    id: 29,
    govId: "EG-WAD",
    name: "New Valley",
    category: "Governorates",
    info: "The largest governorate, featuring breathtaking oases like Dakhla, Kharga, and Farafra.",
    img: url("new_valley1_fs83vt"),
    gallery: gallery(
      "new_valley2_urczuv", "new_valley3_nw2f6d",
      "new_valley4_fyqvgk", "new_valley5_nv7wdt"
    ),
  },
];

// ─── lookup map: govId → entry  (O(1) access inside EgyptMap) ──────────────
export const placesMap = Object.fromEntries(
  placesData.map((p) => [p.govId, p])
);

// ─── fallback for govIds not yet in the list ─────────────────────────────
export const fallback = (govId, title) => ({
  id: null,
  govId,
  name: title || govId,
  category: "Governorates",
  info: "Information coming soon.",
  img: "",
  gallery: [],
});