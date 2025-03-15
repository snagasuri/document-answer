# Final Deployment Checklist

Based on the MongoDB connection issue we've identified, here's your updated checklist to fix the "failed to load chats" 504 error:

## 1. MongoDB Setup

✅ You've already added MongoDB to your Railway project - Good!

✅ You've identified the internal connection string:
```
mongodb://mongo:bYdzdTaKBPNiGLdKRLesohjsSvxbLhPU@mongodb.railway.internal:27017
```

## 2. Update Environment Variables in Railway

Add these environment variables to your **backend service** in Railway (not in the MongoDB service):

```
MONGODB_URI=mongodb://mongo:bYdzdTaKBPNiGLdKRLesohjsSvxbLhPU@mongodb.railway.internal:27017/rag
MONGODB_DB=rag
```

Notes:
- The `/rag` at the end of the URI specifies the database name
- You can change it to `/test_database` if you prefer
- The internal connection string only works within Railway's network

## 3. Add Redis to Railway (if not already done)

1. Click "New" in your Railway project
2. Select "Redis"
3. Wait for it to provision
4. Get the internal Redis connection string
5. Add to your backend service:
```
REDIS_URL=redis://default:password@redis.railway.internal:6379
```

## 4. Update CORS Settings

Add your Vercel domain to the CORS settings:
```
CORS_ORIGINS=["https://your-vercel-app-domain.vercel.app","http://localhost:3000"]
```

Replace `your-vercel-app-domain.vercel.app` with your actual Vercel domain.

## 5. Verify Other API Keys

Make sure these are set correctly in your Railway backend service:
```
PINECONE_API_KEY=pcsk_42gfPr_PMk216jxNZvoaS66hPYdSwVDKfuiHxJWnro84u2oSg3Dp9zsGnFXE4LjAT1ZqAA
PINECONE_ENVIRONMENT=aped-4627-b74a
PINECONE_INDEX=rag-index-llama
COHERE_API_KEY=nX3XTbgCGBcS5EVA0ke0xDClVKEqQV1GPrncLNiq
OPENROUTER_API_KEY=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
CLERK_SECRET_KEY=sk_test_RnS7uV21jNdtqTX3KytGwbpSt2vIG6CeqhWGGFcXAJ
CLERK_PUBLISHABLE_KEY=pk_test_c3BlY2lhbC1zaGVwaGVyZC03LmNsZXJrLmFjY291bnRzLmRldiQ
```

## 6. Environment Settings

Add these environment settings:
```
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

## 7. Redeploy Your Backend

Click the "Deploy" button in your Railway dashboard for your backend service.

## 8. Testing

1. Wait for deployment to complete
2. Visit your Vercel frontend application
3. Check if you can now load chat sessions

## 9. Troubleshooting

If issues persist:

1. Check Railway logs for specific error messages
2. Verify the health endpoint is working:
   ```
   https://your-railway-backend-url.railway.app/health
   ```
3. Make sure both MongoDB and Redis are running in Railway

---

**Important Note**: The MongoDB test script (`test_mongodb_connection.py`) won't work from your local machine with the internal Railway connection string. If you need to test locally, you'll need to get the external MongoDB connection string from Railway and use that instead.
