from generate_reply import generate_reply


evidence = [
    {
        "customer_message":
            'Seeing "reloading" every few seconds. Current app unusable.',
        "spotify_reply":
            "Can you send us a DM with your account email/username and device make/model? We'll take a look.",
        "similarity": 0.50,
    },
    {
        "customer_message":
            "Spotify pauses every few seconds.",
        "spotify_reply":
            "Can you let us know the device and operating system you're using?",
        "similarity": 0.34,
    },
]


reply = generate_reply(
    customer_message="Spotify keeps buffering every few seconds",
    intent="PLAYBACK_ISSUE",
    evidence=evidence,
)

print("\nGenerated Reply:")
print(reply)