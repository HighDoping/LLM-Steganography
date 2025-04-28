'use client';
import Image from "next/image";


const dataShards = 6; // Number of data shards
const parityShards = 2; // Number of parity shards


// const res = await rs.encode(Uint8Array.from(fileBuffer));
async function digestMessage(message: string): Promise<ArrayBuffer> {
  const encoder = new TextEncoder();
  const data = encoder.encode(message);
  if (!window.crypto || !window.crypto.subtle) {
    throw new Error("Web Crypto API not available in this environment.");
  }
  const hash = await window.crypto.subtle.digest("SHA-256", data);
  return hash;
}

function indicesToBytes(indices: number[], base: number): Uint8Array {
  if (!Array.isArray(indices) || indices.some(i => typeof i !== 'number' || i < 0 || i >= base)) {
    throw new Error(`Invalid indices array for base ${base}. All indices must be numbers between 0 and ${base - 1}.`);
  }

  let byteValues: number[] = [];

  switch (base) {
    case 256:
      // Direct conversion if base is 256
      return new Uint8Array(indices);

    case 2: {
      const base2Alphabet = "01";
      let base2Encoded = indices.map(i => base2Alphabet[i]).join('');
      const paddingLength = (8 - base2Encoded.length % 8) % 8;
      base2Encoded += "0".repeat(paddingLength);

      for (let i = 0; i < base2Encoded.length; i += 8) {
        const byteString = base2Encoded.slice(i, i + 8);
        byteValues.push(parseInt(byteString, 2));
      }
      return new Uint8Array(byteValues);
    }

    case 4: {
      const base4Alphabet = "0123";
      let base4Encoded = indices.map(i => base4Alphabet[i]).join('');
      // Each byte is 8 bits, each base-4 digit is 2 bits. Need 4 digits per byte.
      const paddingLength = (4 - base4Encoded.length % 4) % 4;
      base4Encoded += "0".repeat(paddingLength);

      for (let i = 0; i < base4Encoded.length; i += 4) {
        const n1 = parseInt(base4Encoded[i], 4);
        const n2 = parseInt(base4Encoded[i + 1], 4);
        const n3 = parseInt(base4Encoded[i + 2], 4);
        const n4 = parseInt(base4Encoded[i + 3], 4);
        // Combine 4 * 2 bits = 8 bits
        const byteValue = (n1 << 6) | (n2 << 4) | (n3 << 2) | n4;
        byteValues.push(byteValue);
      }
      return new Uint8Array(byteValues);
    }

    case 8: {
      const base8Alphabet = "01234567";
      let base8Encoded = indices.map(i => base8Alphabet[i]).join('');
      const paddingLength = (3 - base8Encoded.length % 3) % 3;
      base8Encoded += "0".repeat(paddingLength);

      for (let i = 0; i < base8Encoded.length; i += 3) {
        const octalChunk = base8Encoded.slice(i, i + 3);
        const value = parseInt(octalChunk, 8);
        if (value > 255) {
          console.warn(`Warning: Octal chunk "${octalChunk}" (value ${value}) exceeds 255. Potential data loss or error in original Python logic replication for base 8.`);
          throw new Error(`Octal chunk "${octalChunk}" (value ${value}) cannot be represented as a single byte.`);
        }
        byteValues.push(value);
      }
      return new Uint8Array(byteValues);
    }


    case 16: {
      const base16Alphabet = "0123456789ABCDEF";
      let base16Encoded = indices.map(i => base16Alphabet[i]).join('');
      // Pad to ensure an even number of hex characters (2 chars per byte)
      const paddingLength = (2 - base16Encoded.length % 2) % 2;
      base16Encoded += "0".repeat(paddingLength);

      // Manual hex decoding (equivalent to b16decode)
      for (let i = 0; i < base16Encoded.length; i += 2) {
        const hexByte = base16Encoded.slice(i, i + 2);
        byteValues.push(parseInt(hexByte, 16));
      }
      return new Uint8Array(byteValues);
    }

    case 32: {
      // Base32 decoding is complex. This requires a dedicated library
      // or a full manual implementation. Using a placeholder.
      // You would typically use a library like 'hi-base32'.
      console.warn("Base32 decoding requires a dedicated library (e.g., 'hi-base32') or a full implementation. Returning empty array as placeholder.");

      const base32Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
      let base32Encoded = indices.map(i => base32Alphabet[i]).join('');
      const paddingLength = (8 - base32Encoded.length % 8) % 8; // Base32 uses blocks of 8 chars
      base32Encoded += "=".repeat(paddingLength); // Standard Base32 padding

      // Placeholder: Replace with actual Base32 decoding logic
      // Example using a hypothetical library function:
      // try {
      //   const decoder = new Base32Decoder(); // Fictional library
      //   return decoder.decode(base32Encoded);
      // } catch (e) {
      //   throw new Error(`Failed to decode base32 string: ${e.message}`);
      // }

      // Basic manual sketch (incomplete and likely needs refinement for edge cases):
      const base32Lookup = Object.fromEntries(base32Alphabet.split('').map((char, i) => [char, i]));
      let bits = '';
      for (const char of base32Encoded) {
        if (char === '=') break;
        if (base32Lookup[char] === undefined) throw new Error(`Invalid Base32 character: ${char}`);
        bits += base32Lookup[char].toString(2).padStart(5, '0');
      }

      for (let i = 0; i + 8 <= bits.length; i += 8) {
        byteValues.push(parseInt(bits.slice(i, i + 8), 2));
      }
      return new Uint8Array(byteValues); // Note: This manual sketch might be imperfect, especially with padding.

    }

    case 64: {
      const base64Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      let base64Encoded = indices.map(i => base64Alphabet[i]).join('');
      const paddingLength = (4 - base64Encoded.length % 4) % 4;
      base64Encoded += "=".repeat(paddingLength);

      try {
        // Use native atob for base64 decoding
        const binaryString = atob(base64Encoded);
        // Convert binary string (char codes) to Uint8Array
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes;
      } catch (e) {
        if (e instanceof DOMException && e.name === 'InvalidCharacterError') {
          throw new Error(`Invalid Base64 string generated: ${base64Encoded}`);
        } else {
          throw e; // Re-throw other errors
        }
      }
    }

    case 85: {
      // Base85 (Ascii85) decoding is complex and less common natively.
      // Requires a dedicated library or full manual implementation.
      console.warn("Base85 decoding requires a dedicated library or a full implementation. Returning empty array as placeholder.");

      const base85Alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
      let base85Encoded = indices.map(i => base85Alphabet[i]).join('');
      // Python example used '~' padding, standard Ascii85 often doesn't pad explicitly
      // or has specific rules ('z' for 4 null bytes, etc.). Sticking to python example's pad:
      const paddingLength = (5 - base85Encoded.length % 5) % 5;
      base85Encoded += "~".repeat(paddingLength); // Using '~' as per python example

      // Placeholder: Replace with actual Base85 decoding logic
      throw new Error("Base85 decoding not implemented. Use a library.");
      // return new Uint8Array([]); // Placeholder return
    }

    default:
      throw new Error(
        `Invalid base selected: ${base}. Available options: 2, 4, 8, 16, 32, 64, 85, 256`
      );
  }
}


export default function Home() {
  async function handleDecode() {
    // 1. Get input values
    const encodedText = (document.getElementById('encodedText') as HTMLTextAreaElement).value;
    const password = (document.getElementById('password') as HTMLInputElement).value;
    const base = parseInt((document.getElementById('base') as HTMLInputElement).value);
    const charPerIndex = parseInt((document.getElementById('charPerIndex') as HTMLInputElement).value);

    try {
      // 2. Convert encoded text to array based on charPerIndex

      const encodedTextArray = [];
      for (let i = 0; i < encodedText.length; i += charPerIndex) {
        const chunk = encodedText.slice(i, i + charPerIndex);
        encodedTextArray.push(chunk);
      }
      console.log('Encoded Text Array:', encodedTextArray);
      // Convert each chunk to a number based on the base
      //python:
      // indices = [
      //   int(hashlib.sha256(c.encode()).hexdigest(), 16) % base for c in chunks
      //       ]

      const indicesPromises = encodedTextArray.map(async (chunk) => {
        // 1. Calculate the hash for the chunk (await the promise)
        const hashBuffer: ArrayBuffer = await digestMessage(chunk);
        // console.log(`Hash buffer for chunk "${chunk}":`, hashBuffer); // Optional: log buffer

        // 2. Convert the ArrayBuffer hash to a hexadecimal string
        const hashArray = Array.from(new Uint8Array(hashBuffer)); // Convert buffer to byte array
        const hexString = hashArray
          .map((byte) => byte.toString(16).padStart(2, '0'))
          .join('');
        // console.log(`Hex string for chunk "${chunk}":`, hexString); // Optional: log hex string

        // 3. Convert hex string to BigInt, apply modulo, and convert back to Number
        // Use BigInt for intermediate calculation to handle large hash values accurately
        const hashBigInt = BigInt('0x' + hexString); // Prepend '0x' for BigInt hex parsing
        const baseBigInt = BigInt(base); // Convert base to BigInt
        const indexBigInt = hashBigInt % baseBigInt;

        // Convert the final result back to a standard number.
        // Be aware: if base is extremely large, this could still exceed Number.MAX_SAFE_INTEGER
        // but for typical modulo results, it should be fine.
        return Number(indexBigInt);
      });
      const indices: number[] = await Promise.all(indicesPromises);
      console.log('Indices:', indices);
      // Convert indices to bytestream
      // Convert numbers to Uint8Array
      const encodedDataUint8Array = indicesToBytes(indices, base);
      console.log('Encoded Data Uint8Array:', encodedDataUint8Array);

      // 4. Use Reed-Solomon to decode



      // 5. Convert decoded Uint8Array to text
      // TODO: Implement conversion

      // 6. Decrypt text using password
      // TODO: Implement decryption

      // 7. Display result
      const decodedTextArea = document.getElementById('decodedText') as HTMLTextAreaElement;
      decodedTextArea.value = encodedDataUint8Array.toString(); // Placeholder for decoded text
    } catch (error) {
      console.error('Decoding failed:', error);
      alert('Failed to decode text');
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <div className="w-full max-w-2xl space-y-4">
        <div>
          <label htmlFor="encodedText" className="block mb-2">
            Encoded Text:
          </label>
          <textarea
            id="encodedText"
            className="w-full h-32 p-2 border rounded"
            placeholder="Enter your encoded text here..."
          />
        </div>

        <div>
          <label htmlFor="password" className="block mb-2">
            Password:
          </label>
          <input
            type="password"
            id="password"
            className="w-full p-2 border rounded"
            placeholder="Enter password"
          />
        </div>

        <div className="flex gap-4">
          <div className="flex-1">
            <label htmlFor="base" className="block mb-2">
              Base:
            </label>
            <input
              type="number"
              id="base"
              className="w-full p-2 border rounded"
              placeholder="Enter base number"
              defaultValue={16}
            />
          </div>

          <div className="flex-1">
            <label htmlFor="charPerIndex" className="block mb-2">
              Characters per Index:
            </label>
            <input
              type="number"
              id="charPerIndex"
              className="w-full p-2 border rounded"
              placeholder="Enter characters per index"
              defaultValue={8}
            />
          </div>
        </div>
        <button
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          onClick={handleDecode}
        >
          Decode
        </button>

        <div>
          <label htmlFor="decodedText" className="block mb-2">
            Decoded Text:
          </label>
          <textarea
            id="decodedText"
            className="w-full h-32 p-2 border rounded"
            readOnly
            placeholder="Decoded text will appear here..."
          />
        </div>
      </div>
    </main>
  );
}
