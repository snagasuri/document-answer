# Fix for RAG Application 504 Timeout Issue

## Issue Diagnosis

Based on our analysis, your RAG application is experiencing a 504 Gateway Timeout error when attempting to load chat sessions. This is happening because:

1. Your backend deployed on Railway can't connect to MongoDB properly
2. The MongoDB connection string is hardcoded in your `backend/core/config.py` file
3. Now that you've added a MongoDB instance to Railway, you need to update your environment variables to point to this new MongoDB instance

## Files Created to Help

1. `railway-env-variables.txt` - Template of all environment variables your Railway backend needs
2. `railway-setup-guide.md` - Step-by-step guide to configure Railway correctly
3. `test_mongodb_connection.py` - Script to verify your MongoDB connection works

## Solution Steps

1. **Update MongoDB Connection:**
   - Get the connection string from your Railway MongoDB instance
   - Add it as a `MONGODB_URI` environment variable in your Railway backend service
   - This will override the hardcoded connection string in your code

2. **Add Redis for Caching:**
   - If not already done, add a Redis service to your Railway project
   - Add the `REDIS_URL` environment variable in your Railway backend service

3. **Configure CORS Settings:**
   - Update the `CORS_ORIGINS` environment variable to include your Vercel app domain

4. **Verify Other API Keys:**
   - Ensure all other API keys (Pinecone, OpenRouter, Cohere, Clerk) are correctly set

5. **Redeploy Your Backend:**
   - After updating all environment variables, redeploy your backend on Railway

## How to Test

1. Run the MongoDB connection test script:
   ```
   python test_mongodb_connection.py <your-mongodb-uri>
   ```

2. Check the Railway health endpoint after deployment:
   ```
   https://your-railway-app-url.railway.app/health
   ```

3. Test your frontend application to see if chat sessions load correctly

## Common Issues

1. **Collection Creation:** MongoDB collections will be created automatically the first time they're used, so no need to manually initialize them.

2. **MongoDB URI Format:** Make sure your MongoDB URI is in the correct format (`mongodb://username:password@host:port/database`).

3. **Railway Logs:** Check the Railway logs for specific error messages if you still encounter issues.

## Next Steps

1. Follow the detailed instructions in `railway-setup-guide.md`
2. Test your connection with `test_mongodb_connection.py`
3. After updating environment variables and redeploying, visit your Vercel app to verify the fix
