declare module 'vis-timeline/standalone' {
    export class Timeline {
        constructor(container: HTMLElement, items: unknown, options?: unknown);
        on(event: string, callback: (properties: unknown) => void): void;
        destroy(): void;
    }
}

declare module 'vis-data/standalone' {
    export class DataSet<T = unknown> {
        constructor(data?: T[], options?: unknown);
        add(data: T | T[]): void;
        update(data: T | T[]): void;
        remove(id: string | number | (string | number)[]): void;
    }
}
