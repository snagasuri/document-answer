'use client';

import { useUser, useClerk } from '@clerk/nextjs';
import { useState, useRef, useEffect } from 'react';

export interface ChatHeaderProps {
  sessionId: string;
  onToggleDocuments?: () => void;
  showDocuments?: boolean;
}

export function ChatHeader({ sessionId, onToggleDocuments, showDocuments }: ChatHeaderProps) {
  const { user } = useUser();
  const { signOut } = useClerk();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="border-b border-gray-200 p-3">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-medium text-gray-800">chat session</h1>
          <p className="text-xs font-light text-gray-500">session id: {sessionId}</p>
        </div>
        <div className="flex items-center space-x-3">
          {onToggleDocuments && (
            <button
              onClick={onToggleDocuments}
              className="flex items-center text-sm text-gray-600 hover:text-gray-800 transition-colors"
              title={showDocuments ? "hide documents" : "show documents"}
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-5 w-5 mr-1" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={1.5} 
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" 
                />
              </svg>
              {showDocuments ? "hide documents" : "show documents"}
            </button>
          )}
          {user && (
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 hover:bg-gray-100 rounded-md p-1.5 transition-colors"
              >
                <img
                  src={user.imageUrl}
                  alt={user.fullName || 'user'}
                  className="w-7 h-7 rounded-md border border-gray-200"
                />
                <span className="text-sm text-gray-700">{user.fullName}</span>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className={`h-4 w-4 text-gray-500 transition-transform ${showUserMenu ? 'rotate-180' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {showUserMenu && (
                <div className="absolute right-0 mt-1 w-48 bg-white border border-gray-200 rounded-md shadow-sm py-1 z-50">
                  <button
                    onClick={() => signOut()}
                    className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    sign out
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
