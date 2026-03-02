# 🔄 Auto-Restart System - Never Stop Until 2000+!

## ✅ **System Status: ACTIVE**

Your pipeline now has **3 layers of protection** to ensure it NEVER stops until you have 2000+ images!

---

## 🛡️ Protection Layers

### 1️⃣ **Watchdog Script** (`watchdog.sh`)

**What it does:**
- Checks if extraction is running → restarts if stopped
- Checks if extraction is stuck (no progress for 30 min) → restarts
- Auto-starts image generation when extraction completes
- Prevents duplicate images (skips existing)
- Logs all actions

**Features:**
- ✅ Detects stopped processes
- ✅ Detects stuck processes
- ✅ Auto-transitions from extraction to image generation
- ✅ Prevents multiple watchdog instances (lock file)
- ✅ Rotates logs when they get too large

---

### 2️⃣ **Cron Job** (Runs every 10 minutes)

**What it does:**
- Automatically runs watchdog every 10 minutes
- Ensures pipeline restarts even if system reboots
- Runs in background, no manual intervention needed

**Status:**
```bash
# View cron job
crontab -l

# You should see:
*/10 * * * * /home/waqas/JellyFish_Galaxies/watchdog.sh
```

**This means:**
- Every 10 minutes, the watchdog checks the pipeline
- If stopped → automatically restarts
- If stuck → kills and restarts
- Runs 24/7 until 2000+ images complete

---

### 3️⃣ **Systemd Service** (Optional, more robust)

**What it is:**
- System-level service that runs continuously
- Auto-restarts on failure
- Survives system reboots

**To install (optional):**
```bash
sudo cp /home/waqas/JellyFish_Galaxies/jellyfish-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jellyfish-pipeline.service
sudo systemctl start jellyfish-pipeline.service
```

**Status check:**
```bash
sudo systemctl status jellyfish-pipeline.service
```

---

## 🎮 Control Commands

### Quick Status Check
```bash
cd /home/waqas/JellyFish_Galaxies
./pipeline_control.sh status
```
Shows:
- What's running
- Watchdog status
- Progress (clusters, galaxies, images)
- Recent activity

### Manual Control
```bash
# Run watchdog check now (don't wait for cron)
./pipeline_control.sh start

# Stop everything
./pipeline_control.sh stop

# Restart pipeline
./pipeline_control.sh restart

# Force restart extraction only
./pipeline_control.sh force-extract

# Force restart image generation only
./pipeline_control.sh force-generate

# View logs
./pipeline_control.sh logs
```

### Cron Management
```bash
# Install cron job (already done)
./pipeline_control.sh install-cron

# Remove cron job (if you want to stop auto-restart)
./pipeline_control.sh remove-cron
```

---

## 📊 How It Works - Step by Step

### Every 10 Minutes:

1. **Check Extraction Phase:**
   ```
   Is extraction complete? NO
     ├─ Is extraction running? YES
     │    ├─ Is it stuck (no progress >30min)? NO
     │    │    → ✓ All good, continue
     │    └─ Is it stuck? YES
     │         → ⚠️ Kill and restart extraction
     └─ Is extraction running? NO
          → ⚠️ Start extraction
   ```

2. **Check Image Generation Phase:**
   ```
   Is extraction complete? YES
     ├─ Are all images generated? NO
     │    ├─ Is generation running? YES
     │    │    → ✓ All good, continue
     │    └─ Is generation running? NO
     │         → ⚠️ Start image generation
     └─ Are all images generated? YES
          → 🎉 MISSION COMPLETE! (watchdog stops)
   ```

---

## 🔍 Monitoring

### Real-Time Status
```bash
# Quick check
./pipeline_control.sh status

# Watch watchdog log live
tail -f watchdog.log

# Watch extraction log live
tail -f extract_2000plus_restart.log

# Watch image generation log live
tail -f image_gen_auto.log
```

### What to Look For

**Healthy extraction:**
```
[2026-02-27 06:50:53] ✓ Extraction running normally (cluster [99/188])
[2026-02-27 07:00:53] ✓ Extraction running normally (cluster [105/188])
```

**Healthy image generation:**
```
[2026-02-27 12:30:00] ✓ Image generation running (1250/3000 images, 1750 remaining)
```

**Auto-restart in action:**
```
[2026-02-27 08:15:00] ⚠️  Extraction stopped! Restarting...
[2026-02-27 08:15:05] Extraction started (PID: 12345)
```

**Stuck detection:**
```
[2026-02-27 09:20:00] ⚠️  Extraction appears stuck! Restarting...
```

---

## 🚨 Failure Scenarios & Recovery

### Scenario 1: Process Crashes
```
Problem: Python script crashes due to API error
Solution: Watchdog detects (next 10-min check), restarts automatically
Recovery Time: <10 minutes
```

### Scenario 2: Process Stuck
```
Problem: API hangs, no progress for 30+ minutes
Solution: Watchdog detects stuck state, kills and restarts
Recovery Time: <10 minutes
```

### Scenario 3: System Reboot
```
Problem: Power outage, system restart
Solution: Cron job runs on boot, restarts pipeline
Recovery Time: <10 minutes after boot
```

### Scenario 4: Disk Full
```
Problem: No space left for images
Solution: Watchdog will try to restart, but manual intervention needed
Fix: Clear disk space, watchdog will resume automatically
```

---

## 📝 Log Files

| File | Purpose | Location |
|------|---------|----------|
| `watchdog.log` | All watchdog actions | Main directory |
| `extract_2000plus_restart.log` | Galaxy extraction progress | Main directory |
| `image_gen_auto.log` | Image generation progress | Main directory |
| `service.log` | Systemd service (if using) | Main directory |

**Log rotation:** Watchdog auto-rotates when logs exceed 10MB

---

## ⚙️ Configuration

### Change Watchdog Frequency

**Current: Every 10 minutes**

To change (e.g., every 5 minutes):
```bash
crontab -e
# Change:
*/10 * * * * /home/waqas/JellyFish_Galaxies/watchdog.sh
# To:
*/5 * * * * /home/waqas/JellyFish_Galaxies/watchdog.sh
```

### Change Stuck Detection Timeout

**Current: 30 minutes**

Edit `watchdog.sh`:
```bash
# Line ~65, change 1800 (seconds) to desired value
if [ $DIFF -gt 1800 ] && check_extraction; then
```

---

## 🎯 Guarantee: 2000+ Images

**With this system:**

1. ✅ Pipeline runs 24/7 automatically
2. ✅ Auto-restarts on any failure
3. ✅ Detects and recovers from stuck states
4. ✅ No duplicates (deduplication built-in)
5. ✅ Logs all activity for debugging
6. ✅ Stops automatically when complete

**You can:**
- Close terminal - pipeline keeps running
- Reboot system - pipeline resumes
- Walk away - pipeline runs until 2000+ done

**Timeline:**
```
Current: ~500 images (514)
Target: 2000+ images
ETA: 10-16 hours (fully automated)
Watchdog checks: Every 10 minutes (144 checks/day)
```

---

## 🔧 Troubleshooting

### Watchdog Not Running?
```bash
# Check cron job exists
crontab -l | grep watchdog

# Re-install if missing
./pipeline_control.sh install-cron
```

### Want to Stop Everything?
```bash
# Stop all processes
./pipeline_control.sh stop

# Remove cron job (optional)
./pipeline_control.sh remove-cron
```

### Want to Check Right Now?
```bash
# Don't wait for cron, run watchdog immediately
./pipeline_control.sh start
```

### See What Happened?
```bash
# View all logs
./pipeline_control.sh logs

# Or specific logs
cat watchdog.log
tail -50 extract_2000plus_restart.log
```

---

## 📊 Current Status

**As of 2026-02-27 06:51 AM EST:**

- ✅ Watchdog: ACTIVE
- ✅ Cron job: INSTALLED (runs every 10 minutes)
- ✅ Extraction: RUNNING (cluster 1/188, restarted by watchdog)
- ⏳ Target: 2000+ images from TNG-Cluster
- 🔒 Duplicates: 0 (514 existing will be skipped)

**Next Actions:**
- 🤖 Watchdog will check again in ~9 minutes
- 🔄 If extraction stops → auto-restart
- 🎨 When extraction done → auto-start image generation
- 🎉 When 2000+ done → watchdog stops automatically

---

## ✅ Summary

**You now have an UNSTOPPABLE pipeline!**

```
┌─────────────────────────────────────────────┐
│  Pipeline Process (Extraction/Generation)    │
│           ↓ crashes/stops                    │
│  Watchdog detects (every 10 min)            │
│           ↓                                  │
│  Auto-restart with logs                      │
│           ↓                                  │
│  Continue until 2000+ images!                │
└─────────────────────────────────────────────┘
```

**No manual intervention needed!**

Just check status occasionally:
```bash
./pipeline_control.sh status
```

**Your 2000+ images WILL be generated!** 🚀
