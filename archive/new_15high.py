#!/usr/bin/env python3
import argparse
import base64
import os
import sys
from openai import OpenAI


def main():
    # Set up CLI argument parsing with optional flags
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
        dest="reference",
        help="Path to an image file to use as a visual reference.",
        default=None
    )

    args = parser.parse_args()

    # Validate that the reference file exists if specified
    if args.reference and not os.path.isfile(args.reference):
        print(f"Error: Reference file '{args.reference}' does not exist.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    print(f"Prompt:    '{args.prompt}'")
    print(f"Saving to: '{args.output}'")

    try:
        if args.reference:
            print(f"Reference: '{args.reference}'")
            print("Generating with image reference via GPT Image 1.5 edits...")

            with open(args.reference, "rb") as ref_image:
                response = client.images.edit(
                    model="gpt-image-1.5",
                    image=ref_image,
                    prompt=args.prompt,
                    n=1,
                    size="1024x1024",
                )
        else:
            print("Generating new image from scratch with GPT Image 1.5...")
            response = client.images.generate(
                model="gpt-image-1.5",
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
