# Android skills

A mirror of [`device-ai/litert`](https://github.com/android/skills/tree/main/device-ai/litert) in [android/skills](https://github.com/android/skills) at commit [`45fd173`](https://github.com/android/skills/tree/45fd1739a4dbca5452c3e936570797f7f6261cd7/device-ai/litert): the [`litert-runtime`](litert-runtime/) and [`litert-lm`](litert-lm/) skills, byte for byte. Changes to them start here as pull requests and then go to android/skills. [`tests/`](tests/) builds their Kotlin into two apps and runs it on a device.

To sync with a newer commit, run this from the root of this repository with a clone of android/skills at `<android-skills>`; `diff -r` prints nothing when the copy is exact:

    for s in litert-runtime litert-lm; do rsync -a --delete <android-skills>/device-ai/litert/$s/ skills/android/$s/; diff -r <android-skills>/device-ai/litert/$s skills/android/$s; done
