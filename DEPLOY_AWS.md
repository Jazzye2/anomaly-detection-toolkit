# Deploying to AWS (App Runner + optional CloudFront)

The live app at https://anomaly-detection-toolkit.streamlit.app/ is hosted
free on Streamlit Community Cloud, and that's the right choice for this
project, it costs nothing and needed zero extra setup. This guide is for
deploying a second copy on AWS, useful as a demonstration of real cloud
deployment skills, or if you want it running somewhere AWS-native. It isn't
a replacement for the free version.

**Cost, read this first:** unlike Streamlit Cloud, AWS App Runner bills for
compute continuously while the service is running, there is no free tier
scale-to-zero here. At the smallest size (1 vCPU / 2 GB) this is roughly a
few dollars a month if left running 24/7, pause the service in the App
Runner console when you're not actively using it to avoid ongoing charges.

## Why App Runner, not EC2 or Lambda

- **Lambda can't run this at all.** Streamlit needs a persistent process
  holding an open websocket connection for every interaction (the slider,
  row selection). Lambda is stateless and spins up per request, it has no
  way to hold that connection open. This isn't a configuration limitation,
  it's a different execution model entirely.
- **EC2 would work but means you own the server.** OS patching, security
  groups, keeping the Streamlit process alive across reboots (a systemd
  service), all manual.
- **App Runner is managed.** Point it at the GitHub repo, it builds and
  runs the app, restarts it if it crashes, and gives you a URL. No server
  to maintain.

## Step 1: Create the App Runner service

1. AWS Console → search "App Runner" → open the console.
2. Click **Create service**.
3. Source: **Source code repository**. Click **Add new** under GitHub
   connections, this opens GitHub's own authorization screen, approve
   AWS App Runner's access to your repositories (first time only).
4. Repository: `Jazzye2/anomaly-detection-toolkit`. Branch: `master`.
5. Deployment trigger: **Automatic**, so it redeploys on every push, same
   behavior as the Streamlit Cloud version.
6. Build settings: choose **Use a configuration file**. This repo already
   has `apprunner.yaml` at its root, App Runner will read it automatically,
   no manual build/start command entry needed.
7. Service name: something like `anomaly-toolkit`.
8. Size: **1 vCPU, 2 GB memory** is plenty for this app, no reason to pick
   anything larger.
9. Auto scaling: leave the defaults (min 1 instance), this is a demo app,
   it doesn't need to scale out.
10. Review and click **Create & deploy**. First build takes a few minutes.

Once it's done, App Runner gives you a URL like
`https://xxxxxxxxxx.us-east-1.awsapprunner.com`, that's the live app.

## Step 2 (optional): CloudFront in front of it

Worth doing if you want a cleaner custom domain or CDN caching, not
required otherwise. Point CloudFront at the App Runner URL as a custom
origin (not S3, this app isn't static), same pattern as your other
projects' CloudFront setups.

**One real decision here:** if you also apply your existing cookie-gate
tooling (`cloudfront-security`) to this CloudFront distribution, the app
becomes private, only accessible with the cookie. That's the right call
for a staging copy you want to test privately before it's public, but
it defeats the point for the main instance you want to share and post,
don't gate the public-facing one.

## Keeping two live copies in sync

Both the Streamlit Cloud app and an AWS App Runner copy would auto-deploy
from the same GitHub repo on every push, so they stay in sync automatically,
no extra work needed to keep them matching.
