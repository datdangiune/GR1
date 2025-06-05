
import React from 'react';
import { AlertCircle, X } from 'lucide-react';

interface ErrorMessageProps {
  message: string;
  onDismiss: () => void;
  isDark: boolean;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, onDismiss, isDark }) => {
  return (
    <div className={`mb-4 p-3 rounded-lg border-l-4 border-red-500 ${
      isDark ? 'bg-red-900 bg-opacity-20' : 'bg-red-50'
    }`}>
      <div className="flex items-start">
        <AlertCircle className="w-5 h-5 text-red-500 mr-2 mt-0.5" />
        <div className="flex-1">
          <p className={`text-sm ${isDark ? 'text-red-300' : 'text-red-700'}`}>
            {message}
          </p>
        </div>
        <button
          onClick={onDismiss}
          className={`ml-2 ${isDark ? 'text-red-300 hover:text-red-200' : 'text-red-500 hover:text-red-700'}`}
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default ErrorMessage;
