# Railway Environment Setup Guide

Follow these steps to configure your Railway backend deployment with the proper environment variables:

## 1. Get Your MongoDB Connection String

First, you need to get the connection string for your newly added MongoDB instance:

1. Log into your Railway dashboard
2. Go to your project that contains your backend and the new MongoDB instance
3. Click on your MongoDB service
4. Look for the "Connect" tab or button
5. Find the connection string (it should look something like: `mongodb://username:password@host:port/database`)
6. Copy this connection string - you'll need it in the next step

## 2. Set Required Environment Variables

In your Railway dashboard, navigate to your backend service (not the MongoDB service) and add/update these environment variables:

### Essential Variables

```
MONGODB_URI=<paste your MongoDB connection string from step 1>
MONGODB_DB=rag
```

### Redis (If you haven't already added a Redis instance to Railway)

1. Add a Redis service to your Railway project
2. Get the Redis connection string similar to how you got the MongoDB string
3. Set the environment variable:
```
REDIS_URL=<your Redis connection string>
```

### Update CORS Settings

Make sure to update the CORS settings to include your Vercel domain:

```
CORS_ORIGINS=["https://your-vercel-app-domain.vercel.app","http://localhost:3000"]
```
Replace `your-vercel-app-domain.vercel.app` with your actual Vercel domain.

### API Keys (Already in your .env.production)

These should already be configured in your environment, but verify they're present:

```
PINECONE_API_KEY=pcsk_42gfPr_PMk216jxNZvoaS66hPYdSwVDKfuiHxJWnro84u2oSg3Dp9zsGnFXE4LjAT1ZqAA
PINECONE_ENVIRONMENT=aped-4627-b74a
PINECONE_INDEX=rag-index-llama
OPENROUTER_API_KEY=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
COHERE_API_KEY=nX3XTbgCGBcS5EVA0ke0xDClVKEqQV1GPrncLNiq
CLERK_SECRET_KEY=sk_test_RnS7uV21jNdtqTX3KytGwbpSt2vIG6CeqhWGGFcXAJ
CLERK_PUBLISHABLE_KEY=pk_test_c3BlY2lhbC1zaGVwaGVyZC03LmNsZXJrLmFjY291bnRzLmRldiQ
```

### Other Important Variables

```
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

## 3. Redeploy Your Backend

After setting all the environment variables:

1. Click the "Deploy" button in your Railway dashboard for your backend service
2. Wait for the deployment to complete
3. Check the logs for any errors

## 4. Test Your Application

After redeployment:

1. Visit your Vercel frontend application
2. Log in and check if chat sessions load correctly now
3. If you still experience issues, check the Railway logs for errors

## 5. Troubleshooting

If you still encounter issues:

1. Check the Railway logs for any specific error messages
2. Test the backend health endpoint: `https://your-railway-app.railway.app/health`
3. Ensure you have both MongoDB and Redis properly configured in Railway
4. Verify that the MongoDB connection string is correct and includes the database name
