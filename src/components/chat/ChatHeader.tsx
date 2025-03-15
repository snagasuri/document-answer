'use client';

import { useUser } from '@clerk/nextjs';

export interface ChatHeaderProps {
  sessionId: string;
  onToggleDocuments?: () => void;
  showDocuments?: boolean;
}

export function ChatHeader({ sessionId, onToggleDocuments, showDocuments }: ChatHeaderProps) {
  const { user } = useUser();

  return (
    <div className="macos-header">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="macos-title">chat session</h1>
          <p className="macos-caption">session id: {sessionId}</p>
        </div>
        <div className="flex items-center space-x-3">
          {onToggleDocuments && (
            <button
              onClick={onToggleDocuments}
              className="flex items-center text-sm text-gray-600 hover:text-gray-800"
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
            <div className="flex items-center space-x-2">
              <img
                src={user.imageUrl}
                alt={user.fullName || 'user'}
                className="user-profile-pic"
                style={{ width: '28px', height: '28px', borderRadius: '6px', border: '1px solid #e5e7eb' }}
              />
              <span className="macos-subtitle">{user.fullName}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
