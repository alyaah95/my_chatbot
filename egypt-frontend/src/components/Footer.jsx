import React from 'react';

function Footer() {
  return (
    <footer>
      <div className="footer-pattern">𓂀</div>
      <div className="container">
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Egyptia. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;