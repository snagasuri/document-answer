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
    <div className="flex h-screen w-full bg-white">
      {/* Chat Sidebar */}
      <ChatSidebar activeChatId={sessionId} />

      <div className="flex flex-1 bg-white">
        {/* Document Panel */}
        {showDocumentPanel && (
          <div className="w-80 border-r border-gray-200 bg-gray-50 overflow-y-auto">
            <div className="flex justify-between items-center p-3">
              <h2 className="text-sm font-medium text-gray-800">documents</h2>
              <button 
                onClick={() => setShowDocumentPanel(false)}
                className="text-gray-500 hover:text-gray-700 transition-colors"
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
        <div className="flex flex-col flex-1">
          <ChatHeader 
            sessionId={sessionId} 
            onToggleDocuments={() => setShowDocumentPanel(!showDocumentPanel)}
            showDocuments={showDocumentPanel}
          />
          <div className="flex-1 overflow-y-auto p-4">
            <ChatList messages={messages} isLoading={isLoading} />
          </div>
          <div className="border-t border-gray-200">
            <div className="bg-gray-50">
              <ContextWindow 
                usedTokens={contextWindow?.usedTokens || 0}
                maxTokens={contextWindow?.maxTokens || 128000}
                promptTokens={tokenUsage?.promptTokens || 0}
                completionTokens={tokenUsage?.completionTokens || 0}
              />
            </div>
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
