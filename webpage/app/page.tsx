'use client';
import Image from "next/image";
import { ReedSolomon } from '@bnb-chain/reed-solomon';


const dataShards = 6; // Number of data shards
const parityShards = 2; // Number of parity shards
const segmentSize = 16777216; // Size of each segment in bytes

const rs = new ReedSolomon(dataShards, parityShards, segmentSize);

// const res = await rs.encode(Uint8Array.from(fileBuffer));
async function digestMessage(message: string): Promise<ArrayBuffer> {
  const encoder = new TextEncoder();
  const data = encoder.encode(message);
  const hash = await window.crypto.subtle.digest("SHA-256", data);
  return hash;
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


      const indices = encodedTextArray.map((chunk) => {
        // Convert chunk to a number using SHA-256 hash
        digestMessage(chunk).then((hash) =>
          console.log('Hash:', hash),
        );
        return parseInt(hash.toString(), 16) % base;
      });
      console.log('Indices:', indices);
      // Convert indices to bytestream
      const numbers = indices.map((index) => {
        // Convert index to a number
        return index;
      });
      // Convert numbers to Uint8Array
      const encodedDataUint8Array = new Uint8Array(numbers);

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
