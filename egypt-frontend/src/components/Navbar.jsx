import React from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
  const [currentPage, setCurrentPage] = React.useState('home');
  return (
    <header>
      <div className="container">
        <nav>
          <Link to="/" style={{ textDecoration: 'none' }}><div className="logo" style={{cursor: 'pointer'}}>
            🇪🇬 Egypt Info
          </div></Link>
          <ul className="nav-links">
            <li>
              <Link to="/">
              <button className={currentPage === 'home' ? 'active' : ''}
                onClick={() => setCurrentPage('home')}>
                Home
              </button>
              </Link>
            </li>
            <li>
              <Link to="/assistant">
              <button 
                onClick={() => setCurrentPage('assistant')} 
                className={currentPage === 'assistant' ? 'active' : ''}>
                Assistant
              </button>
              </Link>
            </li>
            <li>
              <Link to="/history">
              <button 
                onClick={() => setCurrentPage('history')} 
                className={currentPage === 'history' ? 'active' : ''}>
                History
              </button>
              </Link>
            </li>
            <li>
              <Link to="/geography">
              <button 
                onClick={() => setCurrentPage('geography')} 
                className={currentPage === 'geography' ? 'active' : ''}>
                Geography
              </button>
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}

export default Navbar;