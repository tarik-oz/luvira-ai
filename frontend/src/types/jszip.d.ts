declare module 'jszip' {
  export interface JSZipObject {
    async(type: 'string'): Promise<string>
    async(type: 'uint8array'): Promise<Uint8Array>
    async(type: 'arraybuffer'): Promise<ArrayBuffer>
  }

  export interface JSZipFolder {
    forEach(callback: (relativePath: string, file: JSZipObject) => void): void
    file(name: string): JSZipObject | null
  }

  export interface JSZipInstance {
    file(name: string): JSZipObject | null
    folder(name: string): JSZipFolder | null
  }

  interface JSZipStatic {
    loadAsync(data: ArrayBuffer | Uint8Array): Promise<JSZipInstance>
  }

  const JSZip: JSZipStatic
  export default JSZip
}
