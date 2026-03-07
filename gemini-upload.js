/**
 * Client-side utilities for uploading media to the Gemini Files API.
 *
 * Handles the entire pipeline in the browser:
 *   1. Download file from Google Drive (user's access token)
 *   2. Upload to Gemini Files API (multipart/related)
 *   3. Poll until file state is ACTIVE
 *
 * This avoids Vercel serverless function timeouts by keeping
 * heavy I/O (Drive download + Gemini upload) on the client.
 */

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com";

/**
 * Download a file from Google Drive.
 * @returns {Promise<Blob>}
 */
export async function downloadFromDrive(accessToken, fileId) {
    const res = await fetch(
        `https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`,
        { headers: { Authorization: `Bearer ${accessToken}` } },
    );
    if (!res.ok) {
        throw new Error(`Falha ao baixar arquivo do Drive (${res.status}).`);
    }
    return res.blob();
}

/**
 * Upload a file to the Gemini Files API using the resumable upload protocol.
 * Works directly from the browser and handles large files reliably.
 *
 * Step 1: Initiate upload (send metadata, get upload URL)
 * Step 2: Upload file data to the upload URL
 *
 * @returns {Promise<object>} The file metadata from Gemini.
 */
export async function uploadToGemini(apiKey, blob, mimeType, displayName) {
    const fileSize = blob.size;

    // --- Step 1: Initiate resumable upload ---
    const initRes = await fetch(
        `${GEMINI_API_BASE}/upload/v1beta/files?key=${apiKey}`,
        {
            method: "POST",
            headers: {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": String(fileSize),
                "X-Goog-Upload-Header-Content-Type": mimeType,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ file: { displayName } }),
        },
    );

    if (!initRes.ok) {
        const errText = await initRes.text();
        throw new Error(
            `Upload para Gemini falhou na inicialização (${initRes.status}): ${errText}`,
        );
    }

    const uploadUrl = initRes.headers.get("X-Goog-Upload-URL");
    if (!uploadUrl) {
        throw new Error("Gemini não retornou URL de upload.");
    }

    // --- Step 2: Upload file data ---
    const uploadRes = await fetch(uploadUrl, {
        method: "POST",
        headers: {
            "Content-Length": String(fileSize),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        body: blob,
    });

    if (!uploadRes.ok) {
        const errText = await uploadRes.text();
        throw new Error(
            `Upload para Gemini falhou no envio (${uploadRes.status}): ${errText}`,
        );
    }

    const data = await uploadRes.json();
    return data.file;
}

/**
 * Poll the Gemini Files API until the file reaches ACTIVE state.
 */
export async function waitForFileActive(apiKey, fileName, maxWaitMs = 180_000) {
    const start = Date.now();
    while (Date.now() - start < maxWaitMs) {
        const res = await fetch(
            `${GEMINI_API_BASE}/v1beta/${fileName}?key=${apiKey}`,
        );
        if (!res.ok) {
            throw new Error(
                `Erro ao verificar status do arquivo (${res.status}).`,
            );
        }
        const file = await res.json();
        if (file.state === "ACTIVE") return file;
        if (file.state === "FAILED") {
            throw new Error(`Processamento do arquivo falhou: ${fileName}`);
        }
        await new Promise((r) => setTimeout(r, 3000));
    }
    throw new Error(`Timeout: arquivo ${fileName} não ficou pronto.`);
}

/**
 * Full pipeline: Download from Drive → Upload to Gemini → Wait for ACTIVE.
 * @param {string} accessToken  Google OAuth2 token (for Drive)
 * @param {string} apiKey       Gemini API key
 * @param {{ id: string, name: string, mimeType: string }} file
 * @param {(status: string) => void} [onStatus] Progress callback
 * @returns {Promise<{ fileUri: string, mimeType: string }>}
 */
export async function prepareFileForGemini(
    accessToken,
    apiKey,
    file,
    onStatus,
) {
    // 1. Download from Google Drive
    onStatus?.(`Baixando "${file.name}" do Drive…`);
    const blob = await downloadFromDrive(accessToken, file.id);

    // 2. Upload to Gemini Files API
    onStatus?.(`Enviando "${file.name}" para o Gemini…`);
    const uploaded = await uploadToGemini(
        apiKey,
        blob,
        file.mimeType,
        file.name,
    );

    // 3. Wait for processing if needed (common for videos)
    if (uploaded.state !== "ACTIVE") {
        onStatus?.(`Processando "${file.name}"…`);
        const active = await waitForFileActive(apiKey, uploaded.name);
        return { fileUri: active.uri, mimeType: active.mimeType };
    }

    return { fileUri: uploaded.uri, mimeType: uploaded.mimeType };
}
