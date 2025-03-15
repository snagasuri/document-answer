'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import { useChat } from '@/hooks/useChat';
import { ChatList } from '@/components/chat/ChatList';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatHeader } from '@/components/chat/ChatHeader';
import { ContextWindow } from '@/components/chat/ContextWindow';
import DocumentUpload from '@/components/chat/DocumentUpload';
import DocumentList from '@/components/chat/DocumentList';
import { ChatSidebar } from '@/components/chat/ChatSidebar';

export default function ChatSessionPage() {
  const params = useParams();
  const { user } = useUser();
  const sessionId = params.sessionId as string;
  const [showDocumentPanel, setShowDocumentPanel] = useState(false);
  const [documentsProcessing, setDocumentsProcessing] = useState(false);
  
  const { 
    messages, 
    sendMessage, 
    isLoading, 
    tokenUsage,
    contextWindow
  } = useChat(sessionId);

  const handleDocumentUploadComplete = (documentId: string, filename: string) => {
    // Refresh document list or show a notification
    console.log(`Document uploaded: ${filename} (${documentId})`);
  };

  const handleDocumentDelete = (documentId: string) => {
    // Handle document deletion
    console.log(`Document deleted: ${documentId}`);
  };

  return (
    <div className="chat-session-container">
      {/* Chat Sidebar */}
      <ChatSidebar activeChatId={sessionId} />

      <div className="flex flex-1 bg-white">
        {/* Document Panel */}
        {showDocumentPanel && (
          <div className="document-panel">
            <div className="document-panel-header">
              <h2 className="document-panel-title">documents</h2>
              <button 
                onClick={() => setShowDocumentPanel(false)}
                className="document-panel-close"
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-5 w-5" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={1.5} 
                    d="M6 18L18 6M6 6l12 12" 
                  />
                </svg>
              </button>
            </div>
            <DocumentUpload 
              sessionId={sessionId} 
              onUploadComplete={handleDocumentUploadComplete} 
            />
            <DocumentList 
              sessionId={sessionId} 
              onDeleteDocument={handleDocumentDelete}
              onProcessingStatusChange={(isProcessing) => setDocumentsProcessing(isProcessing)}
            />
          </div>
        )}

        {/* Chat Area */}
        <div className="chat-area">
          <ChatHeader 
            sessionId={sessionId} 
            onToggleDocuments={() => setShowDocumentPanel(!showDocumentPanel)}
            showDocuments={showDocumentPanel}
          />
          <div className="chat-messages">
            <ChatList messages={messages} isLoading={isLoading} />
          </div>
          <div className="chat-input-container">
            <ContextWindow 
              usedTokens={contextWindow?.usedTokens || 0}
              maxTokens={contextWindow?.maxTokens || 128000}
              promptTokens={tokenUsage?.promptTokens || 0}
              completionTokens={tokenUsage?.completionTokens || 0}
              sources={messages.length > 0 && messages[messages.length - 1].role === 'assistant' 
                ? messages[messages.length - 1].sources 
                : undefined}
            />
            <ChatInput 
              onSend={sendMessage} 
              isLoading={isLoading} 
              documentsProcessing={documentsProcessing}
              sessionId={sessionId}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
