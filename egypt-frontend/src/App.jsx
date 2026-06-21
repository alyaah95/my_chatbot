import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Assistant from './pages/Assistant';
import History from './pages/History';
import Geography from './pages/Geography';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navbar />

        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/history" element={<History />} />
            <Route path="/geography" element={<Geography />} />
            <Route path="/assistant" element={<Assistant />} />
          </Routes>
        </main>

        <Footer />
      </div>
    </Router>
  );
}

export default App;