def x():
    try:
        return
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    else:
        print("hello")

x()