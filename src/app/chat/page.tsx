'use client';

import { useState } from 'react';
import { ChatSidebar } from '@/components/chat/ChatSidebar';

export default function ChatPage() {
  return (
    <div className="flex h-screen w-full">
      <ChatSidebar />
      
      <div className="flex-1 flex flex-col items-center justify-center bg-gray-50">
        <div className="text-center p-6 max-w-md">
          <h1 className="text-xl font-medium text-gray-800 mb-3">
            welcome to document chat
          </h1>
          <p className="text-sm text-gray-600 mb-5">
            select an existing chat from the sidebar or create a new chat to get started.
          </p>
          <button 
            onClick={() => {
              const newChatButton = document.querySelector<HTMLButtonElement>('[data-testid="new-chat-button"]');
              if (newChatButton) {
                newChatButton.click();
              }
            }}
            className="px-4 py-2 bg-gray-800 text-white text-sm font-medium rounded-md hover:bg-gray-700 transition-colors"
          >
            create new chat
          </button>
        </div>
      </div>
    </div>
  );
}
