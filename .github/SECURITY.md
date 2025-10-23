# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please follow these guidelines:

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisory** (Preferred)
   - Navigate to the repository's Security tab
   - Click "Report a vulnerability"
   - Fill out the advisory form with details

2. **Email**
   - Send details to the repository maintainer
   - Include "SECURITY" in the subject line
   - Provide a detailed description of the vulnerability

### What to Include

When reporting a vulnerability, please include:

- Type of vulnerability (e.g., buffer overflow, SQL injection, XSS, etc.)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it
- Any potential mitigations you've identified

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Status Update**: Within 5 business days
- **Fix Timeline**: Depends on severity and complexity
  - Critical: 7-14 days
  - High: 14-30 days
  - Medium: 30-60 days
  - Low: 60-90 days

### What to Expect

After you submit a vulnerability report:

1. We will acknowledge receipt of your report
2. We will investigate and validate the vulnerability
3. We will work on a fix and coordinate disclosure
4. We will notify you when the fix is released
5. We will credit you in the security advisory (if desired)

## Security Considerations for Drone Operations

### Physical Safety

This project involves autonomous drone operation. Please be aware of these critical safety considerations:

#### Before Flight

- [ ] Test all code in simulation or with commands disabled first
- [ ] Verify battery level is sufficient (>20%)
- [ ] Check for proper WiFi connection to Tello
- [ ] Ensure open space with no obstacles, people, or animals
- [ ] Review local drone regulations and obtain necessary permissions
- [ ] Have emergency stop mechanism ready (keyboard access)

#### During Development

- [ ] Always start with `tello_enable_commands=False`
- [ ] Test depth estimation with webcam before using drone
- [ ] Implement proper error handling for network failures
- [ ] Add timeouts for all drone commands
- [ ] Log all flight commands for debugging

#### Code Safety Features

- Commands are disabled by default in configuration
- Autonomous mode requires explicit confirmation
- Emergency stop function available (`e` key)
- Battery monitoring and low battery warnings
- Automatic landing on connection loss (Tello feature)

### Software Security

#### Dependency Security

We monitor dependencies for known vulnerabilities. To check your installation:

```bash
# Check for vulnerable packages
pip install safety
safety check -r requirements.txt

# Update dependencies
pip install --upgrade -r requirements.txt
```

#### Network Security

**Tello Connection:**

- Tello creates an unencrypted WiFi network
- Anyone in range can potentially intercept communication
- Do not transmit sensitive data over Tello connection
- Be aware that video stream is unencrypted

**Recommendations:**

- Use Tello only in trusted environments
- Do not modify code to transmit personal information via Tello
- Disconnect from Tello WiFi when not in use

#### Model Weights Security

- Only download model weights from trusted sources
- Verify checksums when available
- Be cautious of arbitrary code execution in checkpoint files
- TensorFlow 2.x has protections, but validate sources

### Data Privacy

This software:

- Does NOT collect or transmit user data
- Does NOT require internet connection (except for installation)
- Processes video locally on your device
- Does NOT store video unless explicitly enabled

If you enable video saving (`save_output=True`):

- Video files are stored locally only
- You are responsible for securing saved files
- Be mindful of privacy when recording in public spaces

## Vulnerability Disclosure Policy

### Coordinated Disclosure

We follow coordinated disclosure principles:

1. **Private Disclosure**: Vulnerabilities are kept private during investigation
2. **Fix Development**: We work on a fix while maintaining confidentiality
3. **Testing**: Fix is tested before release
4. **Public Disclosure**: After fix is released, we publish security advisory
5. **Credit**: Reporters are credited (unless they prefer anonymity)

### Embargo Period

We request a 90-day embargo period before public disclosure to allow:

- Time to develop and test fixes
- Time for users to update their installations
- Coordination with downstream projects if affected

### Public Disclosure

After a fix is released, we will:

- Publish a security advisory on GitHub
- Credit the reporter (if permission granted)
- Describe the vulnerability and its impact
- Provide upgrade instructions
- Include CVE identifier if applicable

## Security Best Practices for Users

### Installation

```bash
# Verify repository authenticity
git clone https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet.git
cd dji-tello-collision-avoidance-pydnet

# Check commit signatures (if available)
git log --show-signature

# Install in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Code

```bash
# Start with safe defaults
python webcam_demo.py  # Test without drone first

# Enable drone commands only when ready
python tello_demo.py --enable_commands  # Commands enabled, manual control

# Autonomous mode - USE WITH EXTREME CAUTION
python tello_demo.py --enable_commands --enable_auto_flight
```

### Regular Updates

```bash
# Keep dependencies updated
pip install --upgrade -r requirements.txt

# Pull latest security fixes
git pull origin main
```

## Known Limitations

### Current Limitations

1. **Depth Estimation Accuracy**
   - PyDNet may produce incorrect depth estimates in certain conditions
   - Poor lighting affects accuracy
   - Textureless surfaces may confuse the model
   - Always visually verify depth estimates

2. **Collision Avoidance**
   - Not fail-safe - should not be solely relied upon
   - May not detect thin obstacles (wires, branches)
   - Reaction time depends on inference speed
   - No redundancy or safety margins built in

3. **Network Reliability**
   - Tello WiFi connection can be unstable
   - Video stream may freeze or drop
   - Commands may be delayed or lost
   - No guaranteed delivery of control commands

### Out of Scope

The following are explicitly NOT covered by this security policy:

- Physical damage to drone or property during use
- Violations of local drone regulations
- Misuse for malicious purposes
- Issues arising from modified code
- Problems with third-party dependencies (report to their maintainers)
- Hardware failures or manufacturing defects in Tello drone

## Security Updates

Security updates will be released as:

- Patch versions (2.0.x) for security fixes
- Clearly marked in release notes
- Announced in repository README
- Tagged with 'security' label in releases

## Contact

For security concerns, please use the reporting methods described above rather than public channels.

For general questions about security features, feel free to open a regular GitHub issue.

## Acknowledgments

We thank the security research community for their efforts in keeping open source software secure. Contributors who responsibly disclose vulnerabilities will be acknowledged in our security advisories.

---

**Last Updated**: October 2025
**Policy Version**: 1.0
