import { redirect } from 'next/navigation';
import { auth } from '@clerk/nextjs/server';

// Instead of creating a chat session on the home page, just redirect to the chat page
// This prevents creating multiple chat sessions when the home page loads
export default async function HomePage() {
  const { userId } = await auth();
  
  if (!userId) {
    redirect('/sign-in');
  }

  // Simply redirect to the chat page
  // The user can create a new chat from there
  redirect('/chat');
}
