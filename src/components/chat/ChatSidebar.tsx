'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useApiClient } from '@/lib';

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
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setContextMenuVisible(true);
    setContextMenuPosition({ 
      x: rect.right,
      y: rect.top 
    });
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
    <div className="w-64 h-full border-r border-gray-200 bg-gray-50 flex flex-col">
      <div className="p-3">
        <button 
          onClick={handleNewChat}
          data-testid="new-chat-button"
          className="w-full flex items-center justify-center gap-2 p-2.5 bg-white text-gray-600 border border-gray-200 rounded-md text-sm hover:bg-gray-50 transition-colors shadow-sm"
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
      
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-3 text-center text-sm text-gray-500">loading chats...</div>
        ) : error ? (
          <div className="p-3 text-center text-sm text-gray-500">failed to load chat sessions</div>
        ) : chatSessions.length === 0 ? (
          <div className="p-3 text-center text-sm text-gray-500">no chat sessions yet</div>
        ) : (
          <ul>
            {chatSessions.map(chat => (
              <li 
                key={chat._id}
                onContextMenu={(e) => handleContextMenu(e, chat._id)}
                className={`relative border-b border-gray-200 hover:bg-gray-100 ${chat._id === activeChatId ? 'bg-gray-100' : ''}`}
              >
                <Link 
                  href={`/chat/${chat._id}`}
                  className="block p-3"
                >
                  <div className="text-sm text-gray-700 truncate">
                    {chat.title || `chat ${formatDate(chat.createdAt)}`}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
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
          className="fixed bg-white shadow-lg rounded-md py-1 z-50 w-32 border border-gray-200"
          style={{ 
            left: `${contextMenuPosition.x + 4}px`,
            top: `${contextMenuPosition.y}px`,
            transform: 'translateX(-100%)'  // Position to the left of the click
          }}
        >
          <button 
            onClick={handleDeleteChat}
            className="w-full text-left px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 hover:text-red-700 transition-colors"
          >
            delete chat
          </button>
        </div>
      )}
    </div>
  );
}
