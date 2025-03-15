'use client';

import { useState } from 'react';
import { ChatSidebar } from '@/components/chat/ChatSidebar';

export default function ChatPage() {
  return (
    <div className="chat-container">
      <ChatSidebar />
      
      <div className="chat-main">
        <div className="chat-welcome">
          <h1>welcome to document chat</h1>
          <p>
            select an existing chat from the sidebar or create a new chat to get started.
          </p>
          <button 
            onClick={() => {
              // Find the new chat button in the sidebar
              const newChatButton = document.querySelector<HTMLButtonElement>('.new-chat-button');
              if (newChatButton) {
                // Only click if it exists
                newChatButton.click();
              } else {
                console.error('New chat button not found');
              }
            }}
            className="create-chat-button"
          >
            create new chat
          </button>
        </div>
      </div>
    </div>
  );
}
