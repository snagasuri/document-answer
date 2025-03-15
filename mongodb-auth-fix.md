# Fix MongoDB Authentication Issue

The error in your logs shows a MongoDB authentication failure. There are two ways to fix this:

## Option 1: Update MongoDB URI with correct auth credentials

In Railway, for your backend service:

1. Modify your MONGODB_URI to use the correct username and password
   ```
   MONGODB_URI=mongodb://mongo:bYdzdTaKBPNiGLdKRLesohjsSvxbLhPU@mongodb.railway.internal:27017/rag?authSource=admin
   ```

   The key addition is `?authSource=admin` at the end, which specifies where to authenticate.

## Option 2: Create a user in MongoDB with proper permissions

1. In Railway, click on your MongoDB service
2. Find an option to access MongoDB shell or console
3. Execute these commands to create a user with proper permissions:
   ```
   use admin
   db.createUser(
     {
       user: "mongo",
       pwd: "bYdzdTaKBPNiGLdKRLesohjsSvxbLhPU",
       roles: [
         { role: "readWrite", db: "rag" }
       ]
     }
   )
   ```

4. Try accessing your app again
