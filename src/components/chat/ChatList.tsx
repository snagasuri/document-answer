'use client';

import { useState } from 'react';
import { Message } from '@/types/chat';
import { formatMessageWithCitations } from '@/utils/formatMessageWithCitations';
import { ContextWindow } from './ContextWindow';

interface ChatListProps {
  messages: Message[];
  isLoading: boolean;
}

export function ChatList({ messages, isLoading }: ChatListProps) {
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  return (
    <div className="chat-messages-list">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${
            message.role === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-[80%] ${
              message.role === 'user'
                ? 'macos-message-user'
                : 'macos-message-assistant'
            }`}
          >
            {/* For assistant messages, always check for citations and format if needed */}
            {message.role === 'assistant' 
              ? (
                  <>
                    {console.log('Checking message for citations:', { 
                      messageId: message.id,
                      contentLength: message.content.length,
                      hasSources: message.sources && message.sources.length > 0,
                      sourcesCount: message.sources?.length || 0,
                      isStreaming: message.isStreaming
                    })}
                    
                    {/* Check for citations in the content */}
                    {(() => {
                      const citationPattern = /\[Source (\d+)\]/g;
                      const matches = [...message.content.matchAll(citationPattern)];
                      console.log(`Found ${matches.length} citation patterns in message content`);
                      if (matches.length > 0) {
                        console.log('Citations found:', matches.map(m => m[0]).join(', '));
                      }
                      
                      // If we have citations in the content or sources are available, format with citations
                      if (matches.length > 0 || (message.sources && message.sources.length > 0)) {
                        return formatMessageWithCitations(message.content, message.sources || []);
                      }
                      
                      // Otherwise just return the plain content
                      return message.content;
                    })()}
                  </>
                )
              : message.content
            }
            
            {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
              <div className="mt-2">
                <ContextWindow 
                  usedTokens={0}
                  maxTokens={0}
                  promptTokens={0}
                  completionTokens={0}
                  sources={message.sources}
                  activeCitation={activeCitation}
                />
              </div>
            )}
          </div>
        </div>
      ))}
      
      {isLoading && (
        <div className="flex justify-start">
          <div className="macos-message-assistant">
            thinking...
          </div>
        </div>
      )}
      
      {messages.length === 0 && !isLoading && (
        <div className="flex flex-col items-center justify-center h-full text-gray-500 mt-10">
          <p className="macos-title">no messages yet</p>
          <p className="macos-subtitle mt-1">start a conversation by sending a message below</p>
        </div>
      )}
    </div>
  );
}
