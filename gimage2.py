#!/usr/bin/env python3
import argparse
import base64
import os
import sys
from contextlib import ExitStack
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(
        description="Generate or edit images with OpenAI's GPT Image API."
    )
    parser.add_argument(
        "prompt", 
        help="Text prompt describing the desired image or edits."
    )
    parser.add_argument(
        "output", 
        help="Path where the output image will be saved (e.g., output.png)."
    )
    parser.add_argument(
        "-r", "--reference",
        dest="references",
        action="append",      # Allows multiple -r flags
        default=[],
        help="Path to an image file reference (can be specified multiple times)."
    )

    args = parser.parse_args()

    # Validate that every reference file exists
    for ref_path in args.references:
        if not os.path.isfile(ref_path):
            print(f"Error: Reference file '{ref_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

    client = OpenAI()

    print(f"Prompt:    '{args.prompt}'")
    print(f"Saving to: '{args.output}'")

    try:
        if args.references:
            print(f"Using {len(args.references)} reference image(s): {args.references}")
            print("Generating with image references via GPT Image 1.5 edits...")

            with ExitStack() as stack:
                # Open all files and ensure they close automatically when done
                open_files = [stack.enter_context(open(path, "rb")) for path in args.references]

                response = client.images.edit(
                    model="gpt-image-1.5",
                    image=open_files if len(open_files) > 1 else open_files[0],
                    prompt=args.prompt,
                    n=1,
                    size="1024x1024",
                )
        else:
            print("Generating new image from scratch with GPT Image 1.5...")
            response = client.images.generate(
                model="gpt-image-2",
                prompt=args.prompt,
                n=1,
                size="1024x1024",
                quality="high",
            )

        # Extract base64 payload from the response
        image_base64 = response.data[0].b64_json

        if image_base64:
            with open(args.output, "wb") as f:
                f.write(base64.b64decode(image_base64))
            print(f"Success! Saved to {args.output}")
        else:
            print("Error: No image data returned from API.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"\nAPI Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
