# Vercel Deployment Guide for MediReminder

## Files Created for Vercel Deployment

1. **vercel.json** - Vercel configuration
2. **app_vercel.py** - Simplified Flask app for serverless deployment
3. **requirements_vercel.txt** - Simplified dependencies
4. **.vercelignore** - Excludes unnecessary files from deployment

## Deployment Steps

### Option 1: Vercel CLI (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: medi-reminder
# - Directory: ./
# - Framework: Other
# - Build command: (leave empty)
# - Output directory: (leave empty)
```

### Option 2: GitHub Integration
1. Push your code to GitHub repository
2. Go to [vercel.com](https://vercel.com)
3. Click "New Project"
4. Import your GitHub repository
5. Configure project settings:
   - Framework Preset: Other
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
6. Click "Deploy"

## Environment Variables (Set in Vercel Dashboard)
```
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
```

## Important Notes

### What's Simplified for Vercel:
- ❌ Removed PaddleOCR (too large for serverless)
- ❌ Removed PyTorch model (too large)
- ❌ Removed WhatsApp automation (requires long-running processes)
- ❌ Removed background scheduler (not supported in serverless)
- ✅ Basic medicine detection using regex
- ✅ User authentication and registration
- ✅ Medicine management
- ✅ Simple dashboard

### Limitations in Vercel Version:
1. **OCR**: Uses placeholder function instead of actual OCR
2. **Medicine Detection**: Uses simple text matching
3. **WhatsApp**: Simulated (no actual sending)
4. **File Storage**: Temporary (files don't persist)
5. **Database**: Uses SQLite (resets on each deployment)

## Production Recommendations

For full functionality, consider these alternatives:

### 1. **Heroku** (Better for this app)
- Supports long-running processes
- Persistent file storage with add-ons
- PostgreSQL database
- Can run OCR and ML models

### 2. **Railway** 
- Similar to Heroku
- Better pricing
- Docker support

### 3. **DigitalOcean App Platform**
- Full Flask app support
- Managed databases
- File storage

### 4. **AWS EC2/Elastic Beanstalk**
- Full control
- Can handle large ML models
- Persistent storage

## For Full Feature Deployment

If you want to deploy with all features intact, use:
1. **Heroku**: Create `Procfile` and `runtime.txt`
2. **Railway**: Use the original `app.py`
3. **DigitalOcean**: Deploy as Docker container

## Vercel Deployment Commands

```bash
# Deploy to production
vercel --prod

# Check deployment status
vercel ls

# View deployment logs
vercel logs [deployment-url]
```

## Troubleshooting Common Vercel Errors

1. **FUNCTION_INVOCATION_TIMEOUT**: Reduce processing time
2. **DEPLOYMENT_NOT_READY**: Wait for build to complete
3. **FUNCTION_PAYLOAD_TOO_LARGE**: Reduce request size
4. **INTERNAL_FUNCTION_INVOCATION_FAILED**: Check logs for errors

This simplified version will deploy successfully to Vercel but with limited functionality compared to your full application.