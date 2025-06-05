
import React from 'react';
import { Heart, Moon, Sun } from 'lucide-react';

interface HeaderProps {
  isDark: boolean;
  toggleTheme: () => void;
}

const Header: React.FC<HeaderProps> = ({ isDark, toggleTheme }) => {
  return (
    <header className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-4 py-3 flex items-center justify-between shadow-sm`}>
      <div className="flex items-center space-x-3">
        <div className="bg-blue-500 p-2 rounded-full">
          <Heart className="w-6 h-6 text-white" />
        </div>
        <h1 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-800'}`}>
          MediBot
        </h1>
      </div>
      
      <button
        onClick={toggleTheme}
        className={`p-2 rounded-full transition-colors ${
          isDark 
            ? 'hover:bg-gray-700 text-gray-300' 
            : 'hover:bg-gray-100 text-gray-600'
        }`}
      >
        {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
      </button>
    </header>
  );
};

export default Header;
