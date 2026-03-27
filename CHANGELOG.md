# Changelog

## 2026-03-27

- VR Mode with Stereoscopic Vision: This feature enables the presentation of simulated environments in immersive stereoscopic vision, significantly enhancing the sense of presence and spatial awareness for users.
- Performance Lag in the XTrainerLeader Module: Resolved an issue causing noticeable lag in the leader arm during operation. The root cause was identified as inefficient thread management in the low-level controller.

## 2026-01-08

- Fixed keyboard teleoperation mesh penetration caused by spring-like gripper behavior: releasing the button instantly opened the gripper, producing excessively large instantaneous displacement (effectively infinite velocity).

## 2025-12-29

- Added bilingual documentation for the `XTrainerVR` WebXR teleoperation workflow in `README.md` and `README.zh.md`.
- Created this changelog to track future documentation and feature updates.
- Updated the headset support tables in both READMEs to mark Quest 3 / PICO 4 as available (✅) while keeping Vision Pro in progress (🔄).
