'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useApiClient } from '@/lib/api-client';

interface ChatSession {
  _id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

interface ChatSidebarProps {
  activeChatId?: string;
}

export function ChatSidebar({ activeChatId }: ChatSidebarProps) {
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [contextMenuVisible, setContextMenuVisible] = useState(false);
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 });
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  
  const router = useRouter();
  const { getChatSessions, createChatSession, deleteChatSession } = useApiClient();

  // Fetch chat sessions
  const fetchChatSessions = async () => {
    try {
      setIsLoading(true);
      const sessions = await getChatSessions();
      
      // Sort sessions by creation date (newest first) and limit to 10
      const sortedSessions = sessions.sort((a, b) => {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      }).slice(0, 10);
      
      setChatSessions(sortedSessions);
    } catch (error) {
      console.error('Failed to fetch chat sessions:', error);
      setError('Failed to load chat sessions');
      // Use empty array as fallback when API fails
      setChatSessions([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Load chat sessions on mount
  useEffect(() => {
    fetchChatSessions();
  }, []);

  // Create a new chat session with debouncing
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  
  const handleNewChat = async () => {
    // Prevent multiple clicks from creating multiple sessions
    if (isCreatingChat) {
      console.log('Already creating a chat session, ignoring click');
      return;
    }
    
    try {
      setIsCreatingChat(true);
      setIsLoading(true);
      const newSession = await createChatSession();
      if (newSession && newSession._id) {
        router.push(`/chat/${newSession._id}`);
      }
    } catch (error) {
      console.error('Failed to create new chat session:', error);
      setError('Failed to create new chat');
    } finally {
      setIsLoading(false);
      // Add a small delay before allowing another chat creation
      setTimeout(() => {
        setIsCreatingChat(false);
      }, 1000);
    }
  };

  // Handle right-click on chat item
  const handleContextMenu = (e: React.MouseEvent, chatId: string) => {
    e.preventDefault();
    setContextMenuVisible(true);
    setContextMenuPosition({ x: e.clientX, y: e.clientY });
    setSelectedChatId(chatId);
  };

  // Handle click outside context menu
  useEffect(() => {
    const handleClickOutside = () => {
      setContextMenuVisible(false);
    };

    if (contextMenuVisible) {
      document.addEventListener('click', handleClickOutside);
    }

    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [contextMenuVisible]);

  // Handle delete chat
  const handleDeleteChat = async () => {
    if (!selectedChatId) return;
    
    try {
      // Call API to delete the chat session
      await deleteChatSession(selectedChatId);
      
      // Remove from state
      setChatSessions(chatSessions.filter(chat => chat._id !== selectedChatId));
      
      // If the active chat was deleted, redirect to the chat list
      if (selectedChatId === activeChatId) {
        router.push('/chat');
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
    } finally {
      setContextMenuVisible(false);
    }
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button 
          onClick={handleNewChat}
          className="new-chat-button"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-4 w-4" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M12 4v16m8-8H4" 
            />
          </svg>
          new chat
        </button>
      </div>
      
      <div className="chat-list">
        {isLoading ? (
          <div className="p-3 text-center">loading chats...</div>
        ) : error ? (
          <div className="p-3 text-center">Failed to load chat sessions</div>
        ) : chatSessions.length === 0 ? (
          <div className="p-3 text-center">no chat sessions yet</div>
        ) : (
          <ul>
            {chatSessions.map(chat => (
              <li 
                key={chat._id}
                onContextMenu={(e) => handleContextMenu(e, chat._id)}
                className={`chat-list-item ${chat._id === activeChatId ? 'active' : ''}`}
              >
                <Link 
                  href={`/chat/${chat._id}`}
                >
                  <div className="chat-list-item-title">
                    {chat.title || `chat ${formatDate(chat.createdAt)}`}
                  </div>
                  <div className="chat-list-item-date">
                    {new Date(chat.updatedAt).toLocaleString()}
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Context Menu */}
      {contextMenuVisible && selectedChatId && (
        <div 
          className="fixed bg-white shadow-macos rounded-md py-1 z-50 w-48 border border-gray-200"
          style={{ 
            left: `${contextMenuPosition.x}px`, 
            top: `${contextMenuPosition.y}px` 
          }}
        >
          <button 
            onClick={handleDeleteChat}
            className="w-full text-left px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100"
          >
            delete chat
          </button>
        </div>
      )}
    </div>
  );
}
